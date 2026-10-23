"""SegNeXt segmentation host (Guo et al., MSCAN conv-attention)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.segmentation.base import ConvDecoder, SegmentationModelBase


class MSCANBlock(nn.Module):
    """Multi-scale convolutional attention (SegNeXt MSCAN block)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.pw_conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.msca = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels),
                nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels),
                nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels),
                nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm(self.pw_conv(self.dw_conv(x))))
        attn = sum(m(x) for m in self.msca) / len(self.msca)
        return residual + x * attn.sigmoid()


class SegNeXtEncoder(nn.Module):
    def __init__(self, dims: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        in_ch = 3
        for dim in dims:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, dim, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dim),
                    MSCANBlock(dim),
                    MSCANBlock(dim),
                )
            )
            in_ch = dim
        self.stages = nn.ModuleList(stages)
        self.out_channels = dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class SegNeXtSegmenter(SegmentationModelBase):
    """SegNeXt MSCAN encoder with lightweight conv-attention decoder."""

    backbone_name = "segnext"

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
        self.encoder = SegNeXtEncoder()
        self.proj = nn.Conv2d(self.encoder.out_channels, decoder_channels, 1)
        self.decoder = ConvDecoder(decoder_channels, decoder_channels, stages=4)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(images))

    def decode(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return self.decoder(features, out_size)


def build_segnext(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    disabled_prior: Optional[str] = None,
) -> SegNeXtSegmenter:
    return SegNeXtSegmenter(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
