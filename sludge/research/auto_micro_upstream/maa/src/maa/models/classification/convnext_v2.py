"""ConvNeXt V2 host for classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.classification.base import ClassificationModelBase

FEATURE_CHANNELS = 384


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block with global response normalisation (GRN)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = nn.Parameter(torch.zeros(1, 1, 1, 4 * dim))
        self.pw2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = x / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = x + self.grn * (x * nx)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtV2(ClassificationModelBase):
    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=dims[-1],
            disabled_prior=disabled_prior,
        )
        self.backbone_name = "convnext_v2"
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
        )
        self.downsample_layers.append(stem)
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            if i > 0:
                self.downsample_layers.append(
                    nn.Conv2d(dims[i - 1], dim, kernel_size=2, stride=2),
                )
            self.stages.append(nn.Sequential(*[ConvNeXtV2Block(dim) for _ in range(depth)]))

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers[0](images)
        for i, stage in enumerate(self.stages):
            if i > 0:
                x = self.downsample_layers[i](x)
            x = stage(x)
        return x


def build_convnext_v2(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    """Build ConvNeXt V2 host with GRN blocks and attention on stage-4 map."""
    del pretrained
    return ConvNeXtV2(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
