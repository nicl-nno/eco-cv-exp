"""Squeeze-and-Excitation channel attention (Hu et al.)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class SEBlock(nn.Module):
    """Residual SE wrapper used as attention adapter on feature maps."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.se = SqueezeExcitation(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(x)
