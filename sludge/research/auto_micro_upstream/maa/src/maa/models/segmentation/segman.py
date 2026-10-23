"""SegMAN segmentation host (segmentation with manifold attention)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.constants import NUM_CLASSES
from maa.models.segmentation.base import ConvDecoder, ConvEncoder, SegmentationModelBase


class ManifoldAttentionBlock(nn.Module):
    """
    SegMAN-style manifold attention: local geometry via depthwise conv gates
    and channel re-weighting on segmentation feature manifolds.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.theta = nn.Conv2d(channels, channels // 8, 1)
        self.phi = nn.Conv2d(channels, channels // 8, 1)
        self.g = nn.Conv2d(channels, channels // 2, 1)
        self.out = nn.Conv2d(channels // 2, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        local = self.local(x)
        theta = self.theta(local).view(b, -1, h * w).permute(0, 2, 1)
        phi = self.phi(local).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(theta, phi) / (theta.shape[-1] ** 0.5), dim=-1)
        g = self.g(local).view(b, -1, h * w)
        y = torch.bmm(g, attn.permute(0, 2, 1)).view(b, -1, h, w)
        return x + self.out(self.act(self.norm(y)))


class SegMANSegmenter(SegmentationModelBase):
    """
    SegMAN encoder-decoder with manifold attention blocks.
    Attention variants (SE/SA/CBAM/MAA) are injected at fusion via base class.
    """

    backbone_name = "segman"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        decoder_channels: int = 256,
        depth: int = 4,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            disabled_prior=disabled_prior,
        )
        self.encoder = ConvEncoder(base=64)
        self.manifold = nn.Sequential(
            *[ManifoldAttentionBlock(self.encoder.out_channels) for _ in range(depth)]
        )
        self.proj = nn.Conv2d(self.encoder.out_channels, decoder_channels, 1)
        self.decoder = ConvDecoder(decoder_channels, decoder_channels, stages=4)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images)
        x = self.manifold(x)
        return self.proj(x)

    def decode(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return self.decoder(features, out_size)


def build_segman(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    disabled_prior: Optional[str] = None,
) -> SegMANSegmenter:
    return SegMANSegmenter(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
