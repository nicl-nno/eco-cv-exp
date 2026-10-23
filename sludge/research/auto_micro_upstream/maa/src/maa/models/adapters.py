"""Adapters that inject SE/SA/CBAM/MAA into host backbone feature streams."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.attention.maa.block import MAAConfig, MicroscopyAwareAttention
from maa.attention.registry import build_feature_attention, is_maa_variant
from maa.attention.maa.block import build_maa_for_ablation


class FeatureMapAttentionAdapter(nn.Module):
    """Applies channel/spatial attention on (B, C, H, W) maps."""

    def __init__(self, attention: str, channels: int, **kwargs) -> None:
        super().__init__()
        if is_maa_variant(attention):
            raise ValueError("Use TokenAttentionAdapter for MAA")
        self.attn = build_feature_attention(attention, channels, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class TokensFromFeatureMap(nn.Module):
    """Project CNN/ViT spatial features to tokens for MAA."""

    def __init__(self, in_channels: int, dim: int, grid: int = 16) -> None:
        super().__init__()
        self.grid = grid
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = F.adaptive_avg_pool2d(feats, (self.grid, self.grid))
        x = self.proj(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = self.norm(tokens)
        return tokens, h, w


class TokensToFeatureMap(nn.Module):
    def __init__(self, dim: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, out_channels)

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = tokens.shape
        x = self.proj(tokens).transpose(1, 2).reshape(b, -1, h, w)
        return x


class TokenAttentionAdapter(nn.Module):
    """
    Feature-map <-> token bridge with Microscopy-Aware Attention.
    Used at fusion points of classification / detection / segmentation hosts.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 256,
        grid: int = 16,
        depth: int = 1,
        disabled_prior: Optional[str] = None,
        **maa_kwargs,
    ) -> None:
        super().__init__()
        self.encode = TokensFromFeatureMap(in_channels, dim, grid=grid)
        blocks = []
        for _ in range(depth):
            if disabled_prior:
                blocks.append(build_maa_for_ablation(dim, disabled_prior=disabled_prior, **maa_kwargs))
            else:
                blocks.append(MicroscopyAwareAttention(MAAConfig(dim=dim, **maa_kwargs)))
        self.blocks = nn.ModuleList(blocks)
        self.decode = TokensToFeatureMap(dim, in_channels)

    def forward(self, feats: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # Resize images to a moderate size for prior extraction efficiency
        if images.shape[-2:] != (512, 512):
            img = F.interpolate(images, size=(512, 512), mode="bilinear", align_corners=False)
        else:
            img = images
        tokens, h, w = self.encode(feats)
        for block in self.blocks:
            tokens = block(tokens, img, tokens_h=h, tokens_w=w)
        out = self.decode(tokens, h, w)
        if out.shape[-2:] != feats.shape[-2:]:
            out = F.interpolate(out, size=feats.shape[-2:], mode="bilinear", align_corners=False)
        return feats + out


def build_attention_adapter(
    attention: str,
    channels: int,
    *,
    maa_dim: int = 256,
    maa_grid: int = 16,
    disabled_prior: Optional[str] = None,
    **kwargs,
) -> nn.Module:
    if is_maa_variant(attention) or attention.lower().startswith("maa"):
        return TokenAttentionAdapter(
            channels,
            dim=maa_dim,
            grid=maa_grid,
            disabled_prior=disabled_prior,
            **kwargs,
        )
    return FeatureMapAttentionAdapter(attention, channels, **kwargs)
