"""Boundary-Aware Attention prior."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.attention.common import outer_product_bias, pool_map_to_tokens, rgb_to_gray


class LearnableEdgeDetector(nn.Module):
    """Lightweight f_edge(I) producing a single-channel edge map."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class FixedSobelEdge(nn.Module):
    """Fixed Sobel magnitude edge operator."""

    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(images)
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


class BoundaryPrior(nn.Module):
    """
    E = σ(f_edge(I));  e_i = mean_{P_i} E;  B_ij = e_i e_j
    """

    def __init__(self, learnable_edge: bool = True, in_ch: int = 3) -> None:
        super().__init__()
        self.edge = LearnableEdgeDetector(in_ch) if learnable_edge else FixedSobelEdge()
        self.learnable_edge = learnable_edge

    def edge_map(self, images: torch.Tensor) -> torch.Tensor:
        raw = self.edge(images)
        return torch.sigmoid(raw)

    def token_scores(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        e_map = self.edge_map(images)
        return pool_map_to_tokens(e_map, num_tokens, tokens_h, tokens_w)

    def forward(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        scores = self.token_scores(images, num_tokens, tokens_h, tokens_w)
        return outer_product_bias(scores)
