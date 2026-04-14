"""Upload trained artifacts to HuggingFace Hub."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from jinja2 import Environment, FileSystemLoader


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "TBD"
    rows = [f"| {k} | {v} |" for k, v in metrics.items()]
    return "| Metric | Value |\n|---|---|\n" + "\n".join(rows)


def render_model_card(template_path: Path, metrics: dict, out_path: Path, **extra: str) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=True,
    )
    tpl = env.get_template(template_path.name)
    out_path.write_text(tpl.render(metrics_table=_format_metrics(metrics), **extra))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="kiselyovd/brain-mri-segmentation")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument("--template", default="docs/model_card.md.j2")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir}")

    metrics = {}
    metrics_path = Path(args.metrics)
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for item in artifacts_dir.rglob("*"):
            if item.is_file():
                rel = item.relative_to(artifacts_dir)
                (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
                (tmp_path / rel).write_bytes(item.read_bytes())
        render_model_card(
            Path(args.template),
            metrics,
            tmp_path / "README.md",
            model_description="Production-grade binary brain-tumor MRI segmentation (LGG / TCGA).",
            github_url="https://github.com/kiselyovd/brain-mri-segmentation",
        )
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(repo_id=args.repo_id, exist_ok=True)
        commit_message = f"Release {args.tag}" if args.tag else "Upload artifacts"
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(tmp_path),
            commit_message=commit_message,
        )
    print(f"Published to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
