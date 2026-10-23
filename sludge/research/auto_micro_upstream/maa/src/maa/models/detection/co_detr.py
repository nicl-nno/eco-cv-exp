"""Co-DETR collaborative hybrid detection host (Zhang et al., 2023)."""

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


class CoHeadBranch(nn.Module):
    """Auxiliary one-stage head branch in Co-DETR collaborative training."""

    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = DetectionHead(channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(x)


class CoDETRDetector(DetectionModelBase):
    """
    Co-DETR with parallel one-stage and transformer branches sharing an FPN
    neck. Primary head drives inference; auxiliary branch adds
    collaborative training signal.
    """

    backbone_name = "co_detr"

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
        self.backbone = MultiScaleStem(base_channels=64)
        self.neck = FPNNeck(self.backbone.out_channels, fpn_channels)
        self.coarse_proj = ConvBNAct(fpn_channels, fpn_channels, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fpn_channels,
            nhead=8,
            dim_feedforward=fpn_channels * 4,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.det_head = DetectionHead(fpn_channels, num_classes)
        self.aux_head = CoHeadBranch(fpn_channels, num_classes)

    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(images)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.forward_backbone(images))
        fused = self.coarse_proj(feats[1])
        b, c, h, w = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)
        tokens = self.transformer_encoder(tokens)
        fused = tokens.transpose(1, 2).reshape(b, c, h, w)
        return self.apply_fusion_attention(fused, images)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict:
        fused = self.forward_features(images)
        cls_logits, box_deltas = self.det_head(fused)
        if self.training and targets is not None:
            aux_cls, aux_box = self.aux_head(fused)
            losses = self.compute_losses(cls_logits, box_deltas, targets)
            aux_losses = self.compute_losses(aux_cls, aux_box, targets)
            losses["loss_aux_cls"] = aux_losses["loss_cls"]
            losses["loss_aux_box"] = aux_losses["loss_box"]
            losses["loss_total"] = losses["loss_total"] + aux_losses["loss_total"]
            return losses
        return self.decode_predictions(cls_logits, box_deltas).as_dict()


def build_co_detr(
    *,
    attention: str = "none",
    num_classes: int = len(DETECTION_CLASS_NAMES),
    disabled_prior: Optional[str] = None,
) -> CoDETRDetector:
    return CoDETRDetector(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
