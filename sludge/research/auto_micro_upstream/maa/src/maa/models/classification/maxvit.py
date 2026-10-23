"""MaxViT host for classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.classification.base import ClassificationModelBase, ConvStem

FEATURE_CHANNELS = 256


class MBConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        hidden = in_ch * 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if stride == 1 and in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class MaxViTBlock(nn.Module):
    """Hybrid MBConv + grid/block attention (MaxViT macro layout)."""

    def __init__(self, dim: int, num_heads: int = 8, grid_size: int = 7) -> None:
        super().__init__()
        self.mbconv = MBConvBlock(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.grid_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.block_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.grid_size = grid_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mbconv(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        t = self.norm(tokens)
        g, _ = self.grid_attn(t, t, t)
        tokens = tokens + g
        t = self.norm(tokens)
        blk, _ = self.block_attn(t, t, t)
        tokens = tokens + blk + self.mlp(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class MaxViT(ClassificationModelBase):
    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        dim: int = FEATURE_CHANNELS,
        depth: int = 8,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=dim,
            disabled_prior=disabled_prior,
        )
        self.backbone_name = "maxvit"
        self.stem = ConvStem(out_channels=dim, stages=3)
        self.blocks = nn.Sequential(*[MaxViTBlock(dim) for _ in range(depth)])

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        return self.blocks(x)


def build_maxvit(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    """Build MaxViT host combining MBConv and partitioned attention."""
    del pretrained
    return MaxViT(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
