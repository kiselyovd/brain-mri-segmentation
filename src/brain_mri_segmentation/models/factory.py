"""Model factory — returns a torch.nn.Module by name."""
from __future__ import annotations

from torch import nn

def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "segformer_b2":
        from transformers import SegformerForSemanticSegmentation

        return SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    if name == "unet":
        from torchvision.models.segmentation import deeplabv3_resnet50

        return deeplabv3_resnet50(num_classes=num_classes, weights_backbone=None)
    raise ValueError(f"Unknown model: {name}")
