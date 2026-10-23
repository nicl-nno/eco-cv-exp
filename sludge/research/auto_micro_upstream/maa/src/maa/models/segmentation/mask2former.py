"""Mask2Former segmentation host (Cheng et al., 2022)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.segmentation.base import ConvDecoder, ConvEncoder, SegmentationModelBase


class PixelDecoder(nn.Module):
    """Multi-scale pixel decoder fusing encoder stages."""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.Conv2d(in_channels, out_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.lateral(x))


class MaskedAttentionHead(nn.Module):
    """Transformer decoder with masked attention over pixel features."""

    def __init__(self, dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.queries = nn.Embedding(num_queries, dim)
        layer = nn.TransformerDecoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=3)

    def forward(self, pixel_features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = pixel_features.shape
        memory = pixel_features.flatten(2).transpose(1, 2)
        q = self.queries.weight.unsqueeze(0).expand(b, -1, -1)
        out = self.decoder(q, memory)
        fused = out.mean(dim=1, keepdim=True).transpose(1, 2).reshape(b, c, 1, 1)
        return torch.nn.functional.interpolate(fused, size=(h, w), mode="bilinear", align_corners=False)


class Mask2FormerSegmenter(SegmentationModelBase):
    """Mask2Former universal segmentation with masked-attention decoder."""

    backbone_name = "mask2former"

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
        self.pixel_decoder = PixelDecoder(self.encoder.out_channels, decoder_channels)
        self.masked_decoder = MaskedAttentionHead(decoder_channels)
        self.decoder = ConvDecoder(decoder_channels, decoder_channels, stages=3)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)
        feats = self.pixel_decoder(feats)
        return self.masked_decoder(feats)

    def decode(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return self.decoder(features, out_size)


def build_mask2former(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    disabled_prior: Optional[str] = None,
) -> Mask2FormerSegmenter:
    return Mask2FormerSegmenter(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
