"""DINOv3 host for classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.classification.base import ClassificationModelBase, PatchTransformerEncoder

FEATURE_CHANNELS = 768


class DINOv3(ClassificationModelBase):
    """ViT backbone for DINO-style self-distillation hosts."""

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        embed_dim: int = FEATURE_CHANNELS,
        depth: int = 12,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=embed_dim,
            disabled_prior=disabled_prior,
        )
        self.backbone_name = "dinov3"
        self.encoder = PatchTransformerEncoder(
            embed_dim=embed_dim, depth=depth, num_heads=12, patch_size=16
        )
        self.register_buffer("center", torch.zeros(1, embed_dim))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)


def build_dinov3(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    """Build DINOv3 ViT host."""
    del pretrained
    return DINOv3(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
