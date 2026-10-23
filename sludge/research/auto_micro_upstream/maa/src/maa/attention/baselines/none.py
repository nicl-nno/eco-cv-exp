"""Identity / no-attention baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class NoAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
