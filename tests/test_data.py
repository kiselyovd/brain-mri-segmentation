"""Data layer smoke tests."""
from __future__ import annotations

from brain_mri_segmentation.data import ImageDataModule


def test_datamodule_instantiates(sample_data_dir):
    dm = ImageDataModule(data_dir=str(sample_data_dir), batch_size=2, num_workers=0)
    assert dm.hparams.batch_size == 2
