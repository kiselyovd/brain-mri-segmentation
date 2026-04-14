"""Lightning module wrappers."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn, optim
from torchmetrics.segmentation import DiceScore, MeanIoU


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss() if num_classes > 1 else nn.BCEWithLogitsLoss()
        self.dice = DiceScore(num_classes=num_classes)
        self.iou = MeanIoU(num_classes=num_classes)
        self.save_hyperparameters(ignore=["model"])

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        logits = out.logits if hasattr(out, "logits") else out
        if logits.shape[-2:] != x.shape[-2:]:
            logits = nn.functional.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self._forward_logits(x)
        loss = self.criterion(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self._forward_logits(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1) if logits.ndim == 4 else (logits.sigmoid() > 0.5).long()
        self.dice(preds, y)
        self.iou(preds, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dice", self.dice, prog_bar=True)
        self.log("val/iou", self.iou, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]
