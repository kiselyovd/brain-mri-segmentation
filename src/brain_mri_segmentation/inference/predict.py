"""Inference CLI — load a checkpoint and predict a binary mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def load_model(checkpoint_path: str | Path):
    """Load a Lightning module from checkpoint, rebuilding the backbone from hparams."""
    import torch

    from ..models import SegmentationModule, build_model

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    model_name = hp.get("model_name")
    num_classes = hp.get("num_classes")
    if model_name is None or num_classes is None:
        raise ValueError(
            "Checkpoint missing model_name/num_classes hparams — re-train after upgrade."
        )
    backbone = build_model(model_name, num_classes=num_classes, pretrained=False)
    return SegmentationModule.load_from_checkpoint(str(checkpoint_path), model=backbone)


def predict(model, input_path: str | Path, image_size: int = 256) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import v2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.train(False)

    img = Image.open(input_path).convert("RGB")
    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    img_t = torch.from_numpy(img_arr).permute(2, 0, 1)
    x = v2.Resize((image_size, image_size), antialias=True)(img_t).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model._forward_logits(x)
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(int)
    return {"mask": mask.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    configure_logging()
    model = load_model(args.checkpoint)
    result = predict(model, args.input)
    print(json.dumps({"mask_shape": [len(result["mask"]), len(result["mask"][0])]}, indent=2))


if __name__ == "__main__":
    main()
