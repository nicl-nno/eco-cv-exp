"""Shared utilities for token-level prior pooling from spatial maps."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def outer_product_bias(scores: torch.Tensor) -> torch.Tensor:
    """B_ij = s_i * s_j  with scores shaped (B, N) -> (B, N, N)."""
    return scores.unsqueeze(-1) * scores.unsqueeze(-2)


def similarity_bias(features: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """B_ij = exp(-(r_i - r_j)^2 / sigma^2) for features (B, N)."""
    diff = features.unsqueeze(-1) - features.unsqueeze(-2)
    return torch.exp(-(diff ** 2) / (sigma ** 2 + 1e-8))


def pairwise_distance_bias(coords: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """B_ij = exp(-||c_i - c_j||^2 / sigma^2), coords (B, N, 2) or (N, 2)."""
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    # (B, N, 1, 2) - (B, 1, N, 2)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist2 = (diff ** 2).sum(dim=-1)
    return torch.exp(-dist2 / (sigma ** 2 + 1e-8))


def build_token_grid(num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Normalized 2D coordinates for a square token grid (N, 2)."""
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        # Non-square token count: place tokens along a unit interval
        ys = torch.linspace(0, 1, num_tokens, device=device, dtype=dtype)
        xs = torch.zeros_like(ys)
        return torch.stack([xs, ys], dim=-1)
    gy, gx = torch.meshgrid(
        torch.linspace(0, 1, side, device=device, dtype=dtype),
        torch.linspace(0, 1, side, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


def pool_map_to_tokens(
    spatial_map: torch.Tensor,
    num_tokens: int,
    tokens_h: Optional[int] = None,
    tokens_w: Optional[int] = None,
) -> torch.Tensor:
    """Average-pool (B, 1, H, W) or (B, H, W) map into (B, N) token scores."""
    if spatial_map.dim() == 3:
        spatial_map = spatial_map.unsqueeze(1)
    b, _, h, w = spatial_map.shape
    if tokens_h is None or tokens_w is None:
        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            tokens_h = tokens_w = side
        else:
            tokens_h, tokens_w = 1, num_tokens
    pooled = F.adaptive_avg_pool2d(spatial_map, (tokens_h, tokens_w))
    return pooled.flatten(2).squeeze(1)  # (B, N)


def rgb_to_gray(images: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (B, 1, H, W) luminance."""
    if images.size(1) == 1:
        return images
    r, g, b = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


class SoftmaxAttention(nn.Module):
    """Standard scaled dot-product attention head (single or multi-head wrapper)."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, C)
            bias: optional (B, N, N) or (B, 1, N, N) added to logits before softmax
        Returns:
            out (B, N, C), attn (B, heads, N, N)
        """
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        logits = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)
            logits = logits + bias
        if attn_mask is not None:
            logits = logits + attn_mask
        attn = self.drop(logits.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out), attn
