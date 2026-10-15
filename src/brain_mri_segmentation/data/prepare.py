"""Patient-level train/val/test split for the LGG Brain MRI dataset."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

from ..utils import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")


def file_sha256(path: Path) -> str:
    """Hash a source file without retaining its payload in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _collect_patients(raw: Path) -> list[Path]:
    return sorted(p for p in raw.iterdir() if p.is_dir() and p.name.startswith("TCGA_"))


def build_split_manifest(raw: Path, *, seed: int, val_frac: float, test_frac: float) -> dict:
    """Validate paired files and record the exact deterministic patient split."""
    if not 0 < val_frac < 1 or not 0 < test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one")
    patients = _collect_patients(raw)
    if not patients:
        raise ValueError(f"No TCGA_* patient dirs under {raw}")
    random.Random(seed).shuffle(patients)
    n_val = max(1, round(len(patients) * val_frac))
    n_test = max(1, round(len(patients) * test_frac))
    n_train = len(patients) - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough patients for three nonempty splits")
    splits = {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val :],
    }
    records = []
    basenames: set[str] = set()
    for split, members in splits.items():
        for patient in sorted(members):
            files = sorted(patient.glob("*.tif"))
            images = [p for p in files if not p.stem.endswith("_mask")]
            masks = {p.name for p in files if p.stem.endswith("_mask")}
            expected = {f"{p.stem}_mask.tif" for p in images}
            if not images or masks != expected:
                raise ValueError("Patient has empty, missing or orphan image/mask pairs")
            for image in images:
                if image.name in basenames:
                    raise ValueError("Duplicate image basename across patients")
                basenames.add(image.name)
                mask = patient / f"{image.stem}_mask.tif"
                records.append(
                    {
                        "split": split,
                        "patient": patient.name,
                        "image": image.relative_to(raw).as_posix(),
                        "mask": mask.relative_to(raw).as_posix(),
                        "image_sha256": file_sha256(image),
                        "mask_sha256": file_sha256(mask),
                    }
                )
    records.sort(key=lambda r: (r["split"], r["patient"], r["image"]))
    serialized = json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n"
    return {
        "schema_version": 1,
        "status": "reconstructed_from_original_source",
        "algorithm": "sorted patients; Python Random(seed).shuffle; train, val, test",
        "seed": seed,
        "val_fraction": val_frac,
        "test_fraction": test_frac,
        "patient_counts": {s: len(v) for s, v in splits.items()},
        "image_counts": {s: sum(r["split"] == s for r in records) for s in SPLITS},
        "records_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "records": records,
    }


def prepare_data(
    raw_dir: Path | str,
    processed_dir: Path | str,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> None:
    raw = Path(raw_dir)
    out = Path(processed_dir)
    log.info("prepare_data.start", raw=str(raw), out=str(out))

    if out.exists() and any(p.name != ".gitkeep" for p in out.iterdir()):
        raise ValueError("Processed destination must be empty; stale files could leak patients")
    manifest = build_split_manifest(raw, seed=seed, val_frac=val_frac, test_frac=test_frac)
    for record in manifest["records"]:
        for kind, directory in (("image", "images"), ("mask", "masks")):
            source = raw / record[kind]
            destination = out / record["split"] / directory / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    (out / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    log.info(
        "prepare_data.done",
        patients=manifest["patient_counts"],
        images=manifest["image_counts"],
        records_sha256=manifest["records_sha256"],
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    prepare_data(args.raw, args.out, seed=args.seed)
