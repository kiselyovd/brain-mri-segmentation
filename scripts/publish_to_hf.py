"""Upload trained artifacts to HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from artifact_manifest import digest, local_path, render, validate
from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "TBD"
    scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    if not scalar:
        return "TBD"
    rows = [f"| {k} | {v} |" for k, v in scalar.items()]
    return "| Metric | Value |\n|---|---|\n" + "\n".join(rows)


def _build_tags(domain_tag: str, extra_csv: str, library_name: str) -> list[str]:
    """Combine domain tag, library name, comma-separated extras into a sorted unique list."""
    tags: set[str] = set()
    if domain_tag:
        tags.add(domain_tag)
    if library_name:
        tags.add(library_name)
    if extra_csv:
        for t in extra_csv.split(","):
            t = t.strip()
            if t:
                tags.add(t)
    if library_name == "transformers":
        tags.add("pytorch")
    return sorted(tags)


def _metric_results_from(
    metrics_json_path: str,
    pipeline_tag: str,
    dataset_name: str,
    dataset_type: str,
) -> list[dict]:
    """Read metrics.json and return model-index metric_results structure."""
    path = Path(metrics_json_path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    numeric = [{"type": k, "value": v} for k, v in raw.items() if isinstance(v, (int, float))]
    if not numeric:
        return []
    task_type_map = {
        "image-segmentation": "image-segmentation",
        "image-classification": "image-classification",
        "text-classification": "text-classification",
        "tabular-classification": "tabular-classification",
    }
    task_type = task_type_map.get(pipeline_tag, pipeline_tag or "other")
    return [
        {
            "task_type": task_type,
            "dataset_type": dataset_type or "unknown",
            "dataset_name": dataset_name or "unknown",
            "metrics": numeric,
        }
    ]


def render_model_card(
    template_path: Path,
    metrics: dict,
    out_path: Path,
    **extra,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
    )
    tpl = env.get_template(template_path.name)
    out_path.write_text(
        tpl.render(metrics_table=_format_metrics(metrics), **extra),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish trained artifacts to HuggingFace Hub.")
    parser.add_argument("--repo-id", default="kiselyovd/brain-mri-segmentation")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metrics", default=None)
    parser.add_argument(
        "--prepare-only", type=Path, help="Write the reviewed payload without publishing"
    )
    parser.add_argument("--template", default="docs/model_card.md.j2")
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--widget-sources",
        default=None,
        metavar="DIR",
        help="Directory of widget PNG examples to upload to samples/ in HF repo.",
    )
    parser.add_argument(
        "--base-model",
        default="",
        help="HF base model ID (e.g. nvidia/segformer-b2-...).",
    )
    parser.add_argument(
        "--hf-dataset",
        default="",
        help="HF dataset ID (e.g. user/my-dataset).",
    )
    parser.add_argument(
        "--dataset-name",
        default="",
        help="Human-readable dataset name (defaults to --hf-dataset).",
    )
    parser.add_argument(
        "--domain-tag",
        default="",
        help="Domain tag (e.g. medical, biology).",
    )
    parser.add_argument(
        "--pipeline-tag",
        default="",
        help="HF pipeline tag (e.g. image-segmentation).",
    )
    parser.add_argument(
        "--library-name",
        default="transformers",
        help="Library name pill on HF (default: transformers).",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated extra tags.",
    )
    parser.add_argument(
        "--hf-export",
        default="artifacts/hf_export",
        metavar="DIR",
        help="Directory produced by export_hf_native.py; contents copied to HF repo root.",
    )
    args = parser.parse_args()
    manifest, values = validate(Path.cwd())
    if args.repo_id != manifest["hub"]["repo_id"]:
        raise SystemExit("Target differs from the reviewed artifact manifest")

    artifacts_dir = Path(args.artifacts)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir}")

    metrics: dict = {}
    metrics_path = Path(args.metrics or manifest["evaluation"]["report"])
    if metrics_path.resolve() != Path(manifest["evaluation"]["report"]).resolve():
        raise SystemExit("The reviewed manifest defines the canonical evaluation report")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    widget_examples: list[dict] = []
    if args.widget_sources:
        widget_dir = Path(args.widget_sources)
        if widget_dir.is_dir():
            widget_examples = [
                {
                    "src": f"https://huggingface.co/{args.repo_id}/resolve/main/samples/{p.name}",
                    "example_title": p.stem,
                }
                for p in sorted(widget_dir.glob("*.png"))
            ]

    dataset_name = args.dataset_name or args.hf_dataset

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        hf_export_dir = Path(args.hf_export)
        for artifact in manifest["hub"]["files"]:
            item = local_path(hf_export_dir, artifact["path"])
            if digest(item) != artifact["sha256"]:
                raise SystemExit(f"Export differs from reviewed manifest: {artifact['path']}")
            dest = local_path(tmp_path, artifact["path"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
        metadata_paths = [
            "artifact-manifest.json",
            manifest["evaluation"]["report"],
            manifest["dataset"]["source_receipt"],
            manifest["dataset"]["split_manifest"],
            "reports/provenance/processor-parity.json",
        ]
        report = json.loads(Path(manifest["evaluation"]["report"]).read_text(encoding="utf-8"))
        metadata_paths.append(f"reports/{report['prediction_arrays']['file']}")
        for name in metadata_paths:
            item, dest = local_path(Path.cwd(), name), local_path(tmp_path, name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

        if args.widget_sources:
            widget_dir = Path(args.widget_sources)
            if widget_dir.is_dir():
                samples_dest = tmp_path / "samples"
                samples_dest.mkdir(parents=True, exist_ok=True)
                for png in sorted(widget_dir.glob("*.png")):
                    shutil.copy2(png, samples_dest / png.name)

        render_model_card(
            template_path=Path(args.template),
            metrics=metrics,
            out_path=tmp_path / "README.md",
            model_description=(
                "Production-grade binary brain-tumor MRI segmentation (LGG / TCGA)."
            ),
            github_url="https://github.com/kiselyovd/brain-mri-segmentation",
            repo_id=args.repo_id,
            base_model=args.base_model,
            library_name=args.library_name,
            pipeline_tag=args.pipeline_tag,
            tags=_build_tags(args.domain_tag, args.tags, args.library_name),
            datasets=[args.hf_dataset] if args.hf_dataset else [],
            dataset_name=dataset_name,
            hf_dataset=args.hf_dataset,
            widget_examples=widget_examples,
            metric_results=_metric_results_from(
                str(metrics_path),
                args.pipeline_tag,
                dataset_name,
                args.hf_dataset,
            ),
            artifact_contract=render(manifest, values),
            artifact_revision=manifest["hub"]["revision"],
        )

        if args.prepare_only:
            shutil.copytree(tmp_path, args.prepare_only)
            print(f"Prepared only; no Hub write: {args.prepare_only}")
            return
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        if api.whoami()["name"] != "kiselyovd":
            raise SystemExit("HF owner guard failed: expected kiselyovd")
        before = api.model_info(args.repo_id)
        if before.id != "kiselyovd/brain-mri-segmentation":
            raise SystemExit("Unexpected target model repository")
        commit_message = f"Release {args.tag}" if args.tag else "Upload artifacts"
        commit = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(tmp_path),
            commit_message=commit_message,
            parent_commit=before.sha,
        )
        after = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
        if after.siblings is None:
            raise SystemExit("Published file metadata is unavailable")
        remote = {f.rfilename: f for f in after.siblings}
        for artifact in manifest["hub"]["files"]:
            if artifact["role"] == "weights":
                lfs = remote[artifact["path"]].lfs
                if lfs is None or lfs.sha256 != artifact["sha256"]:
                    raise SystemExit("Published weight hash differs from reviewed artifact")

    print(f"Published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
