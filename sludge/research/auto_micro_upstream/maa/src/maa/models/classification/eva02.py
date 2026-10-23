"""EVA-02 host for classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.classification.base import ClassificationModelBase

FEATURE_CHANNELS = 768


class EVA02Block(nn.Module):
    """ViT block with SwiGLU FFN as used in EVA-02."""

    def __init__(self, dim: int, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * 8 / 3)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.w3 = nn.Linear(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        t = self.norm1(tokens)
        t, _ = self.attn(t, t, t)
        tokens = tokens + t
        t = self.norm2(tokens)
        t = self.w2(nn.functional.silu(self.w1(t)) * self.w3(t))
        tokens = tokens + t
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class EVA02(ClassificationModelBase):
    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        embed_dim: int = FEATURE_CHANNELS,
        depth: int = 12,
        patch_size: int = 14,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=embed_dim,
            disabled_prior=disabled_prior,
        )
        self.backbone_name = "eva02"
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        grid = 512 // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid, embed_dim))
        self.blocks = nn.Sequential(*[EVA02Block(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed[:, : h * w]
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def build_eva02(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    """Build EVA-02 ViT host."""
    del pretrained
    return EVA02(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
