"""Native processor parity against the real dataset/inference tensor transform."""

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor

from brain_mri_segmentation.data.preprocessing import build_hf_processor, image_tensor


@pytest.mark.parametrize("shape", [(256, 256), (91, 173), (401, 289)])
def test_native_float_input_matches_production_transform(tmp_path, shape) -> None:
    pixels = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    build_hf_processor().save_pretrained(tmp_path)
    processor = AutoImageProcessor.from_pretrained(tmp_path, backend="torchvision")
    # Float32 RGB in [0,255] avoids uint8 rounding during tensor resize.
    native = processor(np.asarray(image, dtype=np.float32), return_tensors="pt").pixel_values[0]
    torch.testing.assert_close(native, image_tensor(image), rtol=0, atol=3e-7)
    assert processor.do_normalize is False
    assert processor.do_reduce_labels is False
    assert native.shape == (3, 256, 256)


def test_original_256_tif_pixels_match_native_pil(sample_data_dir) -> None:
    processor = build_hf_processor()
    for path in (sample_data_dir / "images").glob("*.tif"):
        image = Image.open(path).convert("RGB")
        assert image.size == (256, 256)
        native = processor(image, return_tensors="pt").pixel_values[0]
        torch.testing.assert_close(native, image_tensor(image), rtol=0, atol=1e-7)
