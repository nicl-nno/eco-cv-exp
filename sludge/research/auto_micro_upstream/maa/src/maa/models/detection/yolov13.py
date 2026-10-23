"""YOLOv13 detection host with CSP backbone, SPPF, FPN+PAN neck."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import DETECTION_CLASS_NAMES
from maa.models.detection.base import DetectionHead, DetectionModelBase
from maa.models.layers import CSPBottleneck, ConvBNAct, FPN, PAN, SPPF


class YOLOv13Backbone(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        w1, w2, w3, w4 = width, width * 2, width * 4, width * 8
        self.stem = nn.Sequential(
            ConvBNAct(3, w1, 3, stride=2),
            ConvBNAct(w1, w1, 3, stride=1),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(w1, w2, 3, stride=2),
            CSPBottleneck(w2, w2, n=2),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(w2, w3, 3, stride=2),
            CSPBottleneck(w3, w3, n=4),
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(w3, w4, 3, stride=2),
            CSPBottleneck(w4, w4, n=4),
            SPPF(w4, w4),
        )
        self.out_channels = (w2, w3, w4)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return [c2, c3, c4]


class YOLOv13Neck(nn.Module):
    def __init__(self, in_channels: tuple[int, int, int], out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(in_channels, out_channels)
        self.pan = PAN(out_channels, num_levels=3)
        self.refine = nn.ModuleList(
            [CSPBottleneck(out_channels, out_channels, n=1) for _ in range(3)]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        feats = self.fpn(features)
        feats = self.pan(feats)
        return [ref(f) for ref, f in zip(self.refine, feats)]


class YOLOv13Detector(DetectionModelBase):
    """
    YOLOv13-style anchor-free detector with CSP backbone, SPPF, FPN+PAN neck.
    Attention is injected on the fused mid-level map before the detection head.
    """

    backbone_name = "yolov13"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = len(DETECTION_CLASS_NAMES),
        disabled_prior: Optional[str] = None,
        fpn_channels: int = 256,
        width: int = 64,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            disabled_prior=disabled_prior,
        )
        self.backbone = YOLOv13Backbone(width=width)
        self.neck = YOLOv13Neck(self.backbone.out_channels, fpn_channels)
        self.det_head = DetectionHead(fpn_channels, num_classes)

    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(images)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.forward_backbone(images))
        fused = feats[1]
        return self.apply_fusion_attention(fused, images)


def build_yolov13(
    *,
    attention: str = "none",
    num_classes: int = len(DETECTION_CLASS_NAMES),
    disabled_prior: Optional[str] = None,
) -> YOLOv13Detector:
    return YOLOv13Detector(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
