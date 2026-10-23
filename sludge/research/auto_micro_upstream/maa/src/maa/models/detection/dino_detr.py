"""DINO-DETR detection host (Zhang et al., improved DETR denoising)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import DETECTION_CLASS_NAMES
from maa.models.detection.base import (
    DetectionHead,
    DetectionModelBase,
    FPNNeck,
    MultiScaleStem,
)


class DNTransformerDecoder(nn.Module):
    """Contrastive denoising decoder layer (DINO-DETR)."""

    def __init__(self, dim: int = 256, num_queries: int = 300) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, dim)
        self.noisy_label_embed = nn.Embedding(32, dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.memory_proj = nn.Linear(dim, dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        b = memory.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        noise = self.noisy_label_embed.weight[:8].mean(0).view(1, 1, -1)
        queries = queries + noise
        mem = self.memory_proj(memory)
        return self.decoder(queries, mem)


class DINODETRDetector(DetectionModelBase):
    """
    DINO-DETR with denoising training queries and deformable-attention-style
    multi-scale memory.
    """

    backbone_name = "dino_detr"

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
        self.dn_decoder = DNTransformerDecoder(dim=fpn_channels)
        self.det_head = DetectionHead(fpn_channels, num_classes)

    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(images)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.neck(self.forward_backbone(images))
        fused = feats[1]
        b, c, h, w = fused.shape
        memory = fused.flatten(2).transpose(1, 2)
        decoded = self.dn_decoder(memory)
        fused = decoded.mean(dim=1, keepdim=True).transpose(1, 2).reshape(b, c, 1, 1)
        fused = nn.functional.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)
        return self.apply_fusion_attention(fused, images)


def build_dino_detr(
    *,
    attention: str = "none",
    num_classes: int = len(DETECTION_CLASS_NAMES),
    disabled_prior: Optional[str] = None,
) -> DINODETRDetector:
    return DINODETRDetector(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
