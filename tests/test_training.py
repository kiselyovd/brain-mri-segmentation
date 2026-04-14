"""Training smoke — one-epoch fit on data/sample/."""
from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from brain_mri_segmentation.data import SegmentationDataset
from brain_mri_segmentation.models import SegmentationModule, build_model


def test_fit_one_epoch_on_sample(sample_data_dir: Path) -> None:
    torch.manual_seed(0)
    ds = SegmentationDataset(sample_data_dir, image_size=128)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)

    model = build_model("unet_small", num_classes=2, pretrained=False)
    lit = SegmentationModule(model, num_classes=2, lr=1e-3, model_name="unet_small")

    trainer = L.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        accelerator="cpu",
        enable_progress_bar=False,
    )
    trainer.fit(lit, loader, loader)
    assert "train/loss_epoch" in trainer.callback_metrics
