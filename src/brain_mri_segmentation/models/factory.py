"""Model factory — returns a segmentation model by name."""
from __future__ import annotations

from torch import nn


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "segformer_b2":
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
        )
        if pretrained:
            return SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        return SegformerForSemanticSegmentation(config)
    if name == "unet_small":
        from .unet import UNet

        return UNet(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
