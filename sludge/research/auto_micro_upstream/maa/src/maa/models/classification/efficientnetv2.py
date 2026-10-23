"""EfficientNetV2 host for classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.classification.base import (
    ClassificationModelBase,
    ConvStem,
)

FEATURE_CHANNELS = 256


class MBConv(nn.Module):
    """EfficientNetV2 MBConv block with fused inverted-bottleneck layout."""

    def __init__(self, in_ch: int, out_ch: int, *, expand: int, stride: int) -> None:
        super().__init__()
        hidden = in_ch * expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.project(self.depthwise(self.expand(x)))
        if self.use_residual:
            out = out + x
        return out


class EfficientNetV2(ClassificationModelBase):
    """Depthwise-separable MBConv stack for the EfficientNetV2 host."""

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        width_mult: float = 1.0,
    ) -> None:
        stem = ConvStem(out_channels=FEATURE_CHANNELS, stages=5)
        in_ch = stem.out_channels
        blocks: list[nn.Module] = []
        for expand, out_ch, stride in [(4, 192, 1), (4, 256, 2), (6, 384, 1), (6, 512, 2)]:
            out_ch = int(out_ch * width_mult)
            blocks.append(MBConv(in_ch, out_ch, expand=expand, stride=stride))
            in_ch = out_ch
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=in_ch,
            disabled_prior=disabled_prior,
        )
        self.backbone_name = "efficientnetv2"
        self.stem = stem
        self.blocks = nn.Sequential(*blocks)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(images))


def build_efficientnetv2(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    """Build EfficientNetV2 classification host."""
    del pretrained
    return EfficientNetV2(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
