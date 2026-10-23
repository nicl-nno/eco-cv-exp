"""Shared neural building blocks for classification / detection / segmentation hosts."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act(inplace=True) if act is not None else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, 3, stride=stride, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class Residual(nn.Module):
    def __init__(self, fn: nn.Module, drop_path: float = 0.0) -> None:
        super().__init__()
        self.fn = fn
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.fn(x))


class CSPBottleneck(nn.Module):
    """C2f-style split bottleneck used by YOLO-family hosts."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True, expansion: float = 0.5) -> None:
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = ConvBNAct(in_ch, 2 * hidden, 1)
        self.conv2 = ConvBNAct((2 + n) * hidden, out_ch, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(hidden, hidden, 3),
                    ConvBNAct(hidden, hidden, 3),
                )
                for _ in range(n)
            ]
        )
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.conv1(x).chunk(2, dim=1))
        for blk in self.blocks:
            y.append(blk(y[-1]) + (y[-1] if self.shortcut else 0))
        return self.conv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 5) -> None:
        super().__init__()
        hidden = in_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, 1)
        self.conv2 = ConvBNAct(hidden * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class FPN(nn.Module):
    """Top-down feature pyramid with lateral 1x1 projections."""

    def __init__(self, in_channels: Sequence[int], out_channels: int = 256) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])
        self.outputs = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3) for _ in in_channels]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        assert len(features) == len(self.laterals)
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        return [out(lat) for out, lat in zip(self.outputs, laterals)]


class PAN(nn.Module):
    """Bottom-up path aggregation on top of an FPN."""

    def __init__(self, channels: int, num_levels: int) -> None:
        super().__init__()
        self.downs = nn.ModuleList(
            [ConvBNAct(channels, channels, 3, stride=2) for _ in range(num_levels - 1)]
        )
        self.fuse = nn.ModuleList(
            [ConvBNAct(channels, channels, 3) for _ in range(num_levels - 1)]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        outs = [features[0]]
        for i, (down, fuse) in enumerate(zip(self.downs, self.fuse)):
            x = down(outs[-1])
            if x.shape[-2:] != features[i + 1].shape[-2:]:
                x = F.interpolate(x, size=features[i + 1].shape[-2:], mode="nearest")
            outs.append(fuse(x + features[i + 1]))
        return outs


class MultiScaleEncoder(nn.Module):
    """Generic CNN pyramid producing C2/C3/C4/C5-style maps."""

    def __init__(self, widths: Sequence[int] = (64, 128, 256, 512), depth: int = 2) -> None:
        super().__init__()
        stem = [ConvBNAct(3, widths[0], 3, stride=2)]
        stages = []
        in_ch = widths[0]
        for i, w in enumerate(widths):
            blocks = []
            stride = 1 if i == 0 else 2
            blocks.append(CSPBottleneck(in_ch, w, n=depth, shortcut=True))
            if stride == 2:
                blocks.insert(0, ConvBNAct(in_ch, in_ch, 3, stride=2))
            stages.append(nn.Sequential(*blocks))
            in_ch = w
        self.stem = nn.Sequential(*stem)
        self.stages = nn.ModuleList(stages)
        self.out_channels = list(widths)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm1(tgt)
        tgt = tgt + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm2(tgt)
        tgt = tgt + self.cross_attn(h, memory, memory, need_weights=False)[0]
        tgt = tgt + self.mlp(self.norm3(tgt))
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        num_queries: int = 100,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, dim)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(dim, num_heads) for _ in range(depth)]
        )
        self.class_head = nn.Linear(dim, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4),
            nn.Sigmoid(),
        )

    def forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = memory.size(0)
        tgt = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return self.class_head(tgt), self.bbox_head(tgt)


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim: int, max_h: int = 128, max_w: int = 128) -> None:
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4 for 2D sincos PE")
        self.dim = dim
        pe = torch.zeros(dim, max_h, max_w)
        half = dim // 2
        d_model = half
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_w = torch.arange(max_w).float()
        pos_h = torch.arange(max_h).float()
        pe_w = torch.zeros(d_model, max_w)
        pe_h = torch.zeros(d_model, max_h)
        pe_w[0::2] = torch.sin(pos_w.unsqueeze(0) * div.unsqueeze(1))
        pe_w[1::2] = torch.cos(pos_w.unsqueeze(0) * div.unsqueeze(1))
        pe_h[0::2] = torch.sin(pos_h.unsqueeze(0) * div.unsqueeze(1))
        pe_h[1::2] = torch.cos(pos_h.unsqueeze(0) * div.unsqueeze(1))
        pe[:half] = pe_w.unsqueeze(1).expand(-1, max_h, -1)
        pe[half:] = pe_h.unsqueeze(2).expand(-1, -1, max_w)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :, : x.size(2), : x.size(3)]


class MaskedAttentionDecoder(nn.Module):
    """Lightweight mask-classification decoder for segmentation hosts."""

    def __init__(self, dim: int, num_classes: int, num_queries: int = 50, depth: int = 3) -> None:
        super().__init__()
        self.queries = nn.Embedding(num_queries, dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(dim) for _ in range(depth)])
        self.mask_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.class_embed = nn.Linear(dim, num_classes + 1)
        self.pixel_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, memory_tokens: torch.Tensor, pixel_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        memory_tokens: (B, N, C)
        pixel_feat: (B, C, H, W)
        Returns class logits (B, Q, K+1) and mask logits (B, Q, H, W).
        """
        b = memory_tokens.size(0)
        tgt = self.queries.weight.unsqueeze(0).expand(b, -1, -1)
        for layer in self.layers:
            tgt = layer(tgt, memory_tokens)
        cls = self.class_embed(tgt)
        mask_tokens = self.mask_embed(tgt)
        pix = self.pixel_proj(pixel_feat)
        masks = torch.einsum("bqc,bchw->bqhw", mask_tokens, pix)
        return cls, masks
