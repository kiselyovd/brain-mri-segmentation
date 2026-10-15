"""Export the trained SegFormer backbone into HF-native format (safetensors + config.json)."""

from __future__ import annotations

import argparse
from pathlib import Path

from brain_mri_segmentation.data.preprocessing import build_hf_processor
from brain_mri_segmentation.inference.predict import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.ckpt")
    parser.add_argument("--out", default="artifacts/hf_export")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Training input size; never copy upstream normalization into this export.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lit = load_model(args.checkpoint)
    backbone = lit.model
    if not hasattr(backbone, "save_pretrained"):
        raise SystemExit("Backbone is not transformers-compatible; cannot export natively.")

    # Binary brain-tumor segmentation: label the 2 classes so the HF config
    # renders meaningful id2label/label2id instead of null/LABEL_N placeholders.
    cfg = getattr(backbone, "config", None)
    if cfg is not None and getattr(cfg, "num_labels", None) == 2:
        cfg.id2label = {0: "background", 1: "tumor"}
        cfg.label2id = {"background": 0, "tumor": 1}

    backbone.save_pretrained(out)

    build_hf_processor(args.image_size).save_pretrained(out)

    print(f"Exported HF-native model to {out}")


if __name__ == "__main__":
    main()
