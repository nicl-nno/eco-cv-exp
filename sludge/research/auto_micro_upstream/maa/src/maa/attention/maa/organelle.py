"""Organelle-Centric Attention prior."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.attention.common import outer_product_bias, pool_map_to_tokens


class OrganelleDetector(nn.Module):
    """f_org(I): learnable high-contrast organelle-like activation map."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class OrganellePrior(nn.Module):
    """
    O = σ(f_org(I));  o_i = mean_{P_i} O;  B_ij = o_i o_j
    """

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.detector = OrganelleDetector(in_ch)

    def activation_map(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.detector(images))

    def token_scores(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        return pool_map_to_tokens(
            self.activation_map(images), num_tokens, tokens_h, tokens_w
        )

    def forward(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        scores = self.token_scores(images, num_tokens, tokens_h, tokens_w)
        return outer_product_bias(scores)
