"""Model smoke tests (forward pass, output shape)."""
from __future__ import annotations

import torch

from brain_mri_segmentation.models import build_model


def test_unet_small_forward():
    model = build_model("unet_small", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape == (2, 2, 256, 256)


def test_segformer_b2_forward():
    model = build_model("segformer_b2", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    logits = out.logits if hasattr(out, "logits") else out
    assert logits.shape[0] == 2 and logits.shape[1] == 2
    assert logits.shape[2] in (64, 256) and logits.shape[3] in (64, 256)
