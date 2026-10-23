"""InternImage-H detection host (Wang et al., DCNv3 backbone)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.constants import DETECTION_CLASS_NAMES
from maa.models.detection.base import (
    ConvBNAct,
    DetectionHead,
    DetectionModelBase,
    FPNNeck,
)


class DeformableConv2d(nn.Module):
    """Depthwise convolution with a learned offset field (DCNv3-style block)."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.offset = nn.Conv2d(channels, 2 * kernel_size * kernel_size, 3, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.offset(x)
        x = self.pointwise(self.act(self.norm(self.conv(x))))
        return x


class InternImageBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dcn = DeformableConv2d(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dcn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class InternImageHBackbone(nn.Module):
    def __init__(self, base: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, base, 3, s=2),
            ConvBNAct(base, base, 3),
        )
        c2, c3, c4 = base * 2, base * 4, base * 8
        self.stage2 = nn.Sequential(ConvBNAct(base, c2, 3, s=2), *[InternImageBlock(c2) for _ in range(depth)])
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, s=2), *[InternImageBlock(c3) for _ in range(depth)])
        self.stage4 = nn.Sequential(ConvBNAct(c3, c4, 3, s=2), *[InternImageBlock(c4) for _ in range(depth)])
        self.out_channels = (c2, c3, c4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]


class InternImageHDetector(DetectionModelBase):
    """
    InternImage-H large-model detector with DCNv3-style blocks and FPN neck.
    """

    backbone_name = "internimage_h"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = len(DETECTION_CLASS_NAMES),
        disabled_prior: Optional[str] = None,
        fpn_channels: int = 256,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            disabled_prior=disabled_prior,
        )
        self.backbone = InternImageHBackbone()
        self.neck = FPNNeck(self.backbone.out_channels, fpn_channels)
        self.det_head = DetectionHead(fpn_channels, num_classes)

    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(images)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.forward_backbone(images))
        fused = feats[1]
        return self.apply_fusion_attention(fused, images)


def build_internimage_h(
    *,
    attention: str = "none",
    num_classes: int = len(DETECTION_CLASS_NAMES),
    disabled_prior: Optional[str] = None,
) -> InternImageHDetector:
    return InternImageHDetector(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
