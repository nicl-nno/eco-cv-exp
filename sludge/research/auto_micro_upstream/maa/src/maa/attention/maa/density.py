"""Density-Aware Attention prior / cell population."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.attention.common import (
    build_token_grid,
    outer_product_bias,
    pairwise_distance_bias,
    pool_map_to_tokens,
)


class DensityPrior(nn.Module):
    """
    ρ_i = Σ_j exp(-||c_i - c_j||^2 / σ^2);
    B_ij = ρ_i ρ_j

    Optionally modulate token mass by a soft foreground map before KDE.
    """

    def __init__(self, sigma: float = 0.35, use_mass: bool = False, in_ch: int = 3) -> None:
        super().__init__()
        self.sigma = sigma
        self.use_mass = use_mass
        if use_mass:
            self.mass_net = nn.Sequential(
                nn.Conv2d(in_ch, 8, 3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, 1),
            )
        else:
            self.mass_net = None

    def density_scores(
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
        kernel = pairwise_distance_bias(coords, sigma=self.sigma)  # (B, N, N)
        if self.use_mass and self.mass_net is not None:
            mass = torch.sigmoid(self.mass_net(images))
            m = pool_map_to_tokens(mass, num_tokens, tokens_h, tokens_w)
            # Weighted KDE: ρ_i = Σ_j K_ij m_j
            rho = torch.einsum("bij,bj->bi", kernel, m)
        else:
            rho = kernel.sum(dim=-1)
        # Normalize per batch element
        rho = rho / (rho.mean(dim=-1, keepdim=True) + 1e-6)
        return rho

    def forward(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        rho = self.density_scores(images, num_tokens, tokens_h, tokens_w, coords)
        return outer_product_bias(rho)
