"""CBAM: Convolutional Block Attention Module."""

from __future__ import annotations

import torch
import torch.nn as nn

from maa.attention.baselines.sa import SpatialAttention
from maa.attention.baselines.se import SqueezeExcitation


class ChannelAttentionCBAM(nn.Module):
    """CBAM channel branch uses both avg- and max-pooled descriptors."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.mlp(self.avg_pool(x).view(b, c))
        mx = self.mlp(self.max_pool(x).view(b, c))
        w = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttentionCBAM(channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.cbam = CBAM(channels, reduction, spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(x)
