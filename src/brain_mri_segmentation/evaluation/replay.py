"""Evaluate an identified artifact on a hash-verified patient split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data import SegmentationDataset
from ..data.prepare import file_sha256
from ..inference.predict import load_model
from ..utils import configure_logging, get_logger
from .metrics import METRIC_CONTRACT, binary_confusion, overlap_scores

log = get_logger(__name__)


def verified_test_records(data: Path) -> tuple[dict, list[dict]]:
    """Fail before inference if evaluated files differ from the recorded cohort."""
    manifest = json.loads((data / "split-manifest.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest["records"], sort_keys=True, separators=(",", ":")) + "\n"
    if hashlib.sha256(serialized.encode()).hexdigest() != manifest["records_sha256"]:
        raise ValueError("Split manifest content hash mismatch")
    records = sorted(
        (r for r in manifest["records"] if r["split"] == "test"),
        key=lambda r: Path(r["image"]).name,
    )
    if not records:
        raise ValueError("Test cohort is empty")
    for kind, directory in (("image", "images"), ("mask", "masks")):
        folder = data / "test" / directory
        expected = {Path(r[kind]).name for r in records}
        if expected != {p.name for p in folder.iterdir() if p.is_file()}:
            raise ValueError("Test files differ from split manifest")
        for record in records:
            if file_sha256(folder / Path(record[kind]).name) != record[f"{kind}_sha256"]:
                raise ValueError("Test file content hash mismatch")
    log.info("evaluation.cohort_verified", slices=len(records), hash=manifest["records_sha256"])
    return manifest, records


def patient_bootstrap(patients: list[dict], repeats: int = 2000) -> dict:
    """Seeded patient-cluster bootstrap for pooled overlap; not a model comparison."""
    rng = random.Random(42)
    scores: dict[str, list[float]] = {"dice": [], "iou": []}
    for _ in range(repeats):
        sample = rng.choices(patients, k=len(patients))
        counts = [sum(p["counts"][i] for p in sample) for i in range(4)]
        values = overlap_scores(counts)
        for key in scores:
            scores[key].append(values[key])
    return {
        "method": "patient cluster bootstrap; pooled foreground; percentile 95%",
        "seed": 42,
        "repeats": repeats,
        "intervals": {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in scores.items()},
    }


def identified_model(args: argparse.Namespace) -> tuple[Any, dict]:
    """Load an explicit local checkpoint or pinned public safetensors artifact."""
    if args.checkpoint:
        path = Path(args.checkpoint)
        return load_model(path), {"kind": "lightning", "sha256": file_sha256(path)}
    if not args.revision or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Native HF evaluation requires an immutable 40-hex revision")
    if not args.weights_sha256 or not re.fullmatch(r"[0-9a-f]{64}", args.weights_sha256):
        raise ValueError("Native HF evaluation requires the expected weights SHA256")
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSemanticSegmentation

    weights = Path(
        hf_hub_download(args.hf_model, "model.safetensors", revision=args.revision, token=False)
    )
    if file_sha256(weights) != args.weights_sha256:
        raise ValueError("Native model weights SHA256 mismatch")
    model = AutoModelForSemanticSegmentation.from_pretrained(
        args.hf_model,
        revision=args.revision,
        use_safetensors=True,
        trust_remote_code=False,
        token=False,
    )
    return model, {
        "kind": "native_safetensors",
        "repo_id": args.hf_model,
        "revision": args.revision,
        "sha256": args.weights_sha256,
    }


def main() -> None:
    """Write metrics, per-slice/patient counts, predictions and complete run identities."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--hf-model")
    parser.add_argument("--revision")
    parser.add_argument("--weights-sha256")
    parser.add_argument("--source-revision")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="reports/replay.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.source_revision and not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source-revision must be an immutable 40-hex commit")
    if min(args.batch_size, args.image_size, args.cpu_threads) < 1:
        parser.error("batch-size, image-size and cpu-threads must be positive")

    configure_logging()
    torch.set_num_threads(args.cpu_threads)
    manifest, records = verified_test_records(Path(args.data))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, artifact = identified_model(args)
    model = model.to(device).eval()
    dataset = SegmentationDataset(Path(args.data) / "test", image_size=args.image_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    slices: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    patient_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    with torch.inference_mode():
        for x, target in loader:
            result = model._forward_logits(x.to(device)) if args.checkpoint else model(x.to(device))
            logits = result.logits if hasattr(result, "logits") else result
            logits = torch.nn.functional.interpolate(
                logits, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
            prediction = logits.argmax(dim=1).cpu()
            for p, t in zip(prediction, target, strict=True):
                record = records[len(slices)]
                counts = binary_confusion(p, t).tolist()
                slices.append(
                    {
                        "image": record["image"],
                        "patient": record["patient"],
                        "counts": counts,
                        **overlap_scores(counts),
                    }
                )
                patient_counts[record["patient"]] = [
                    a + b for a, b in zip(patient_counts[record["patient"]], counts, strict=True)
                ]
            predictions.extend(prediction.numpy().astype(np.uint8))
            targets.extend(target.numpy().astype(np.uint8))
            log.info("evaluation.progress", completed=len(slices), total=len(records))
    patients: list[dict[str, Any]] = [
        {"patient": p, "counts": c, **overlap_scores(c)} for p, c in sorted(patient_counts.items())
    ]
    counts = [sum(p["counts"][i] for p in patients) for i in range(4)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = out.with_suffix(".npz")
    np.savez_compressed(arrays_path, predictions=np.stack(predictions), targets=np.stack(targets))
    package_root = Path(__file__).resolve().parents[1]
    source_files = {
        p.relative_to(package_root).as_posix(): file_sha256(p)
        for p in sorted(package_root.rglob("*.py"))
    }
    metrics = {
        "schema_version": 1,
        "status": "replayed_on_reconstructed_original_split",
        "training_membership_verified": False,
        "artifact": artifact,
        "source_revision": args.source_revision,
        "source_files_sha256": source_files,
        "metric_contract": METRIC_CONTRACT,
        "preprocessing": {
            "size": args.image_size,
            "input": "RGB float32/255",
            "resize": "torchvision tensor bilinear antialias",
            "normalize": False,
        },
        "split_records_sha256": manifest["records_sha256"],
        "prediction_arrays": {"file": arrays_path.name, "sha256": file_sha256(arrays_path)},
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "transformers", "torchmetrics", "numpy")
        },
        "device": str(device),
        "test_size": len(slices),
        "test_patients": len(patients),
        "counts_order": ["TP", "FP", "FN", "TN"],
        "counts": counts,
        "global_foreground": overlap_scores(counts),
        "patient_macro": {
            k: sum(p[k] for p in patients) / len(patients)
            for k in ("dice", "iou", "pixel_accuracy")
        },
        "patient_bootstrap": patient_bootstrap(patients),
        "patients": patients,
        "slices": slices,
    }
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    log.info("evaluation.done", slices=len(slices), metrics=metrics["global_foreground"])


if __name__ == "__main__":
    main()
