"""Cell-Compactness Attention prior."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.attention.common import build_token_grid, pairwise_distance_bias, pool_map_to_tokens


class ForegroundEstimator(nn.Module):
    """Optional m_i foreground likelihood modulator."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(images))


class CompactnessPrior(nn.Module):
    """
    B_ij = exp(-||c_i - c_j||^2 / σ^2) * (optional m_i m_j)
    """

    def __init__(self, sigma: float = 0.35, use_foreground: bool = True, in_ch: int = 3) -> None:
        super().__init__()
        self.sigma = sigma
        self.use_foreground = use_foreground
        self.foreground = ForegroundEstimator(in_ch) if use_foreground else None

    def forward(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = images.size(0)
        if coords is None:
            grid = build_token_grid(num_tokens, images.device, images.dtype)
            coords = grid.unsqueeze(0).expand(b, -1, -1)
        bias = pairwise_distance_bias(coords, sigma=self.sigma)
        if self.use_foreground and self.foreground is not None:
            m_map = self.foreground(images)
            m = pool_map_to_tokens(m_map, num_tokens, tokens_h, tokens_w)
            bias = bias * (m.unsqueeze(-1) * m.unsqueeze(-2))
        return bias
