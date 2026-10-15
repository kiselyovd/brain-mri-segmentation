"""Data layer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodule import SegmentationDataModule
    from .dataset import SegmentationDataset

__all__ = ["SegmentationDataModule", "SegmentationDataset"]


def __getattr__(name: str):
    """Keep source-data preparation independent of importing the training runtime."""
    if name == "SegmentationDataModule":
        from .datamodule import SegmentationDataModule

        return SegmentationDataModule
    if name == "SegmentationDataset":
        from .dataset import SegmentationDataset

        return SegmentationDataset
    raise AttributeError(name)
