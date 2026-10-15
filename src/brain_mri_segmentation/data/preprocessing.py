"""The shared RGB tensor preprocessing contract used by training and inference."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2


def image_tensor(image: Image.Image, image_size: int = 256) -> torch.Tensor:
    """Convert RGB to float32/255 before antialiased bilinear tensor resizing."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    return v2.Resize((image_size, image_size), antialias=True)(tensor)


def build_hf_processor(image_size: int = 256):
    """Export native settings; pass RGB float32 arrays to preserve tensor resizing."""
    from transformers import SegformerImageProcessor

    return SegformerImageProcessor(
        size={"height": image_size, "width": image_size},
        do_resize=True,
        resample=2,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=False,
        do_reduce_labels=False,
    )
