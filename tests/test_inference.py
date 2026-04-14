"""End-to-end inference smoke on data/sample/."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import torch

from brain_mri_segmentation.data import SegmentationDataset
from brain_mri_segmentation.inference.predict import load_model, predict
from brain_mri_segmentation.models import SegmentationModule, build_model


def test_predict_on_sample(sample_data_dir: Path, tmp_path: Path) -> None:
    torch.manual_seed(0)
    ds = SegmentationDataset(sample_data_dir, image_size=128)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    model = build_model("unet_small", num_classes=2, pretrained=False)
    lit = SegmentationModule(model, num_classes=2, lr=1e-3, model_name="unet_small")
    trainer = L.Trainer(
        max_epochs=1,
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator="cpu",
    )
    trainer.fit(lit, loader)
    ckpt = tmp_path / "smoke.ckpt"
    trainer.save_checkpoint(str(ckpt))

    reloaded = load_model(ckpt)
    sample_img = next((sample_data_dir / "images").glob("*.tif"))
    result = predict(reloaded, sample_img)
    assert "mask" in result
    mask = result["mask"]
    assert isinstance(mask, list) and mask and len(mask[0]) > 0
    flat = {v for row in mask for v in row}
    assert flat <= {0, 1}
