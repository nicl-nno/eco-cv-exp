"""Mask DINO segmentation host (Li et al., mask classification transformer)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.segmentation.base import ConvDecoder, ConvEncoder, SegmentationModelBase


class MaskDINOTransformer(nn.Module):
    """Mask queries with multi-scale deformable cross-attention."""

    def __init__(self, dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = features.shape
        memory = features.flatten(2).transpose(1, 2)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        q, _ = self.cross_attn(queries, memory, memory)
        queries = queries + q
        q, _ = self.self_attn(queries, queries, queries)
        queries = self.norm(queries + q + self.ffn(queries))
        mask_tokens = queries.mean(dim=1, keepdim=True).transpose(1, 2).reshape(b, c, 1, 1)
        return nn.functional.interpolate(mask_tokens, size=(h, w), mode="bilinear", align_corners=False)


class MaskDINOSegmenter(SegmentationModelBase):
    """
    Mask DINO mask-classification transformer for instance-aware semantic
    segmentation.
    """

    backbone_name = "mask_dino"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            disabled_prior=disabled_prior,
        )
        self.encoder = ConvEncoder(base=64)
        self.proj = nn.Conv2d(self.encoder.out_channels, decoder_channels, 1)
        self.mask_transformer = MaskDINOTransformer(dim=decoder_channels)
        self.decoder = ConvDecoder(decoder_channels, decoder_channels, stages=3)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = self.proj(x)
        return self.mask_transformer(x)

    def decode(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return self.decoder(features, out_size)


def build_mask_dino(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    disabled_prior: Optional[str] = None,
) -> MaskDINOSegmenter:
    return MaskDINOSegmenter(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
