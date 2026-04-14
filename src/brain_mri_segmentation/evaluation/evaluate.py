"""Run model on test set, write reports/metrics.json (Dice + IoU + pixel accuracy)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.segmentation import DiceScore, MeanIoU

from ..data import SegmentationDataset
from ..inference.predict import load_model
from ..utils import configure_logging, get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default="reports/metrics.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint).to(device)
    model.train(False)

    test_ds = SegmentationDataset(Path(args.data) / "test", image_size=args.image_size)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    dice = DiceScore(num_classes=2).to(device)
    iou = MeanIoU(num_classes=2).to(device)
    pixel = BinaryAccuracy().to(device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model._forward_logits(x)
            preds = logits.argmax(dim=1)
            dice.update(preds, y)
            iou.update(preds, y)
            pixel.update(preds, y)

    metrics = {
        "dice": float(dice.compute()),
        "iou": float(iou.compute()),
        "pixel_accuracy": float(pixel.compute()),
        "test_size": len(test_ds),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("done", out=str(out_path), dice=metrics["dice"], iou=metrics["iou"])


if __name__ == "__main__":
    main()
