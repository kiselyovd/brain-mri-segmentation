"""Lightning DataModule for binary brain-MRI segmentation."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .dataset import SegmentationDataset


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 16,
        num_workers: int = 4,
        image_size: int = 256,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: SegmentationDataset | None = None
        self.val_ds: SegmentationDataset | None = None
        self.test_ds: SegmentationDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        root = Path(self.hparams.data_dir)
        self.train_ds = SegmentationDataset(
            root / "train", image_size=self.hparams.image_size, augment=True
        )
        self.val_ds = SegmentationDataset(root / "val", image_size=self.hparams.image_size)
        self.test_ds = SegmentationDataset(root / "test", image_size=self.hparams.image_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )
