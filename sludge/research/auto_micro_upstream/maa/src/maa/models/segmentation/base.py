"""Base class for segmentation hosts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.constants import INPUT_SIZE, NUM_CLASSES
from maa.models.adapters import build_attention_adapter
from maa.attention.registry import is_maa_variant


class SegmentationModelBase(nn.Module, ABC):
    """
    Encoder-decoder / mask-transformer segmentation host.

    Subclasses produce dense logits ``(B, num_classes, H, W)`` aligned with
    input resolution after optional attention on the bottleneck or fusion map.
    """

    backbone_name: str = "segmenter"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        decoder_channels: int = 256,
        disabled_prior: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.attention_name = attention.lower().strip()
        self.num_classes = num_classes
        self.decoder_channels = decoder_channels
        self._uses_maa = is_maa_variant(self.attention_name) or self.attention_name.startswith("maa")

        if self.attention_name in ("none", "no", "identity", "-"):
            self.fusion_attention: nn.Module = nn.Identity()
        else:
            self.fusion_attention = build_attention_adapter(
                self.attention_name,
                decoder_channels,
                disabled_prior=disabled_prior,
            )

        self.seg_head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return bottleneck / fused feature map ``(B, C, H', W')``."""

    @abstractmethod
    def decode(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        """Upsample encoded features toward ``out_size``."""

    def apply_attention(self, feats: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if self._uses_maa:
            return self.fusion_attention(feats, images)
        return self.fusion_attention(feats)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images:
            ``(B, 3, H, W)`` microscopy tensors.

        Returns
        -------
        logits:
            ``(B, num_classes, H, W)`` per-pixel class scores.
        """
        out_size = (images.shape[-2], images.shape[-1])
        feats = self.encode(images)
        feats = self.apply_attention(feats, images)
        decoded = self.decode(feats, out_size)
        logits = self.seg_head(decoded)
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class ConvEncoder(nn.Module):
    """Shared CNN encoder pyramid for segmentation hosts."""

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        dims = [base, base * 2, base * 4, base * 8]
        layers: list[nn.Module] = []
        in_ch = 3
        for dim in dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, dim, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.GELU(),
                    nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.GELU(),
                )
            )
            in_ch = dim
        self.stages = nn.ModuleList(layers)
        self.out_channels = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stages: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(stages):
            next_ch = max(out_channels, ch // 2)
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_ch),
                    nn.GELU(),
                )
            )
            ch = next_ch
        self.layers = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.layers(x)
        if x.shape[-2:] != out_size:
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x
