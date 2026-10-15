"""Explicit foreground overlap contract for integer binary segmentation masks."""

from __future__ import annotations

import torch
from torchmetrics import Metric

METRIC_CONTRACT = {
    "version": "binary-foreground-global-v1",
    "foreground": 1,
    "input": "binary integer masks [B,H,W] after logits upsampling and argmax",
    "aggregation": "sum TP/FP/FN/TN over the entire evaluated cohort before division",
    "empty_target_and_prediction": "Dice=1, IoU=1",
    "empty_target_with_positive_prediction": "Dice=0, IoU=0",
    "patient_macro": "unweighted mean of per-patient pooled foreground overlap",
}


def binary_confusion(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return TP, FP, FN, TN; reject non-binary or mismatched masks."""
    if preds.shape != target.shape or preds.ndim not in (2, 3):
        raise ValueError("Expected matching binary masks with shape [H,W] or [B,H,W]")
    if preds.numel() == 0:
        raise ValueError("Cannot evaluate an empty mask")
    if not torch.all((preds == 0) | (preds == 1)) or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Masks must contain only 0 and 1")
    prediction, truth = preds.bool(), target.bool()
    return torch.stack(
        [
            (prediction & truth).sum(),
            (prediction & ~truth).sum(),
            (~prediction & truth).sum(),
            (~prediction & ~truth).sum(),
        ]
    )


def overlap_scores(counts: list[int]) -> dict[str, float]:
    """Compute foreground scores from pooled TP/FP/FN/TN, with no smoothing."""
    tp, fp, fn, tn = counts
    total = tp + fp + fn + tn
    if total == 0:
        raise ValueError("Cannot evaluate an empty cohort")
    union, dice_denominator = tp + fp + fn, 2 * tp + fp + fn
    return {
        "dice": 2 * tp / dice_denominator if dice_denominator else 1.0,
        "iou": tp / union if union else 1.0,
        "pixel_accuracy": (tp + tn) / total,
    }


class ForegroundOverlap(Metric):
    """Distributed-safe epoch overlap, pooling integer counts across all batches."""

    full_state_update = False
    counts: torch.Tensor

    def __init__(self, score: str) -> None:
        super().__init__()
        if score not in ("dice", "iou"):
            raise ValueError("score must be dice or iou")
        self.score = score
        self.add_state("counts", default=torch.zeros(4, dtype=torch.int64), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate binary confusion counts."""
        self.counts += binary_confusion(preds, target)

    def compute(self) -> torch.Tensor:
        """Calculate one foreground metric after global count aggregation."""
        tp, fp, fn, _ = self.counts.double()
        numerator = 2 * tp if self.score == "dice" else tp
        denominator = numerator + fp + fn
        return torch.where(denominator > 0, numerator / denominator.clamp_min(1), 1.0)
