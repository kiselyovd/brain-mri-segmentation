"""Paired (image, mask) TIF dataset for binary segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .preprocessing import image_tensor


class SegmentationDataset(Dataset):
    def __init__(self, split_dir: Path | str, image_size: int = 256, augment: bool = False) -> None:
        self.root = Path(split_dir)
        self.image_paths = sorted((self.root / "images").glob("*.tif"))
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.root / "masks" / f"{img_path.stem}_mask.tif"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)

        img_t = image_tensor(img, self.image_size)
        mask_t = torch.from_numpy(mask_arr)

        mask_t = v2.Resize(
            (self.image_size, self.image_size),
            interpolation=v2.InterpolationMode.NEAREST,
        )(mask_t.unsqueeze(0)).squeeze(0)
        return img_t, mask_t
