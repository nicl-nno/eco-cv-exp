"""RT-DETR v4 detection host (Baidu, real-time DETR)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import DETECTION_CLASS_NAMES
from maa.models.detection.base import (
    ConvBNAct,
    DetectionHead,
    DetectionModelBase,
    FPNNeck,
    MultiScaleStem,
)


class HybridEncoderBlock(nn.Module):
    """RepVGG-style hybrid encoder block used in RT-DETR family."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = ConvBNAct(channels, channels, 1)
        self.conv3 = ConvBNAct(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv3(self.conv2(self.conv1(x)))


class RTDETRv4Backbone(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.stem = MultiScaleStem(base_channels=channels)
        c1, c2, c3 = self.stem.out_channels
        self.encoder = nn.Sequential(
            HybridEncoderBlock(c3),
            HybridEncoderBlock(c3),
            nn.Conv2d(c3, c3, 1),
        )
        self.out_channels = (c1, c2, c3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = self.stem(x)
        feats[-1] = self.encoder(feats[-1])
        return feats


class RTDETRv4Detector(DetectionModelBase):
    """
    RT-DETR v4 real-time transformer detector.
    CNN hybrid encoder + FPN fusion with attention before IoU-aware head.
    """

    backbone_name = "rt_detr_v4"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = len(DETECTION_CLASS_NAMES),
        disabled_prior: Optional[str] = None,
        fpn_channels: int = 256,
        query_dim: int = 256,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            disabled_prior=disabled_prior,
        )
        self.backbone = RTDETRv4Backbone()
        self.neck = FPNNeck(self.backbone.out_channels, fpn_channels)
        self.query_proj = nn.Linear(fpn_channels, query_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=query_dim,
            nhead=8,
            dim_feedforward=query_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.det_head = DetectionHead(fpn_channels, num_classes)

    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(images)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.forward_backbone(images))
        fused = feats[1]
        b, c, h, w = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = self.transformer(self.query_proj(tokens))
        fused = tokens.transpose(1, 2).reshape(b, self.fpn_channels, h, w)
        return self.apply_fusion_attention(fused, images)


def build_rt_detr_v4(
    *,
    attention: str = "none",
    num_classes: int = len(DETECTION_CLASS_NAMES),
    disabled_prior: Optional[str] = None,
) -> RTDETRv4Detector:
    return RTDETRv4Detector(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
