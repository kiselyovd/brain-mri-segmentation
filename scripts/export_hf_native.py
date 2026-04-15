"""Export the trained SegFormer backbone into HF-native format (safetensors + config.json)."""
from __future__ import annotations

import argparse
from pathlib import Path

from brain_mri_segmentation.inference.predict import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    parser.add_argument("--out", default="artifacts/hf_export")
    parser.add_argument(
        "--base-model",
        default="nvidia/segformer-b2-finetuned-ade-512-512",
        help="HF base model ID to copy preprocessor from.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lit = load_model(args.checkpoint)
    backbone = lit.model
    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit("Backbone is not transformers-compatible; cannot export natively.")
    backbone.save_pretrained(out)

    if args.base_model:
        from transformers import AutoImageProcessor
        AutoImageProcessor.from_pretrained(args.base_model).save_pretrained(out)

    print(f"Exported HF-native model to {out}")


if __name__ == "__main__":
    main()
