"""Swin Transformer V2 host with hierarchical stages and shifted-window attention."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.constants import NUM_CLASSES
from maa.models.classification.base import ClassificationModelBase
from maa.models.layers import DropPath

FEATURE_CHANNELS = 384


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """(B, H, W, C) -> (num_windows*B, window_size, window_size, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """(num_windows*B, window_size, window_size, C) -> (B, H, W, C)."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttentionV2(nn.Module):
    """Multi-head window attention with continuous relative position bias (SwinV2-style)."""

    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        # Continuous relative position bias MLP (log-spaced)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous().float()
        relative_coords[:, :, 0] /= max(window_size - 1, 1)
        relative_coords[:, :, 1] /= max(window_size - 1, 1)
        relative_coords = relative_coords * 8.0
        relative_coords = torch.sign(relative_coords) * torch.log2(torch.abs(relative_coords) + 1.0) / math.log2(8)
        self.register_buffer("relative_coords_table", relative_coords, persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B_, N, C) within windows."""
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Relative position bias
        bias = self.cpb_mlp(self.relative_coords_table).permute(2, 0, 1).unsqueeze(0)  # 1, heads, N, N
        attn = attn + bias
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj(x)


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttentionV2(dim, window_size=window_size, num_heads=num_heads)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm1(x)

        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        if pad_r or pad_b:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted = window_reverse(attn_windows, self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted

        if pad_r or pad_b:
            x = x[:, :h, :w, :].contiguous()
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)

        # MLP on channel-last
        y = x.permute(0, 2, 3, 1)
        y = y + self.drop_path(self.mlp(self.norm2(y)))
        return y.permute(0, 3, 1, 2)


class PatchMerging(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C H W
        b, c, h, w = x.shape
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, w % 2, 0, h % 2))
            _, _, h, w = x.shape
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1).permute(0, 2, 3, 1)
        x = self.reduction(self.norm(x))
        return x.permute(0, 3, 1, 2)


class SwinV2Stage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 8,
        drop_path: float = 0.0,
        downsample: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    drop_path=drop_path,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class SwinV2(ClassificationModelBase):
    """Hierarchical Swin Transformer V2 classification host."""

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        embed_dim: int = 96,
        depths: tuple[int, ...] = (2, 2, 6, 2),
        num_heads: tuple[int, ...] = (3, 6, 12, 24),
        window_size: int = 8,
    ) -> None:
        final_dim = embed_dim * (2 ** (len(depths) - 1))
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=final_dim,
            disabled_prior=disabled_prior,
        )
        self.backbone_name = "swinv2"
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )
        stages = []
        dim = embed_dim
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stages.append(
                SwinV2Stage(
                    dim=dim,
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    downsample=(i < len(depths) - 1),
                )
            )
            if i < len(depths) - 1:
                dim *= 2
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(final_dim)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        for stage in self.stages:
            x = stage(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def build_swinv2(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    del pretrained
    return SwinV2(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
