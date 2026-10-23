"""Focus-Quality Attention prior / Z-axis sharpness."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.attention.common import outer_product_bias, pool_map_to_tokens, rgb_to_gray


class LaplacianSharpness(nn.Module):
    """S = |ΔI| using a fixed 3×3 Laplacian kernel."""

    def __init__(self) -> None:
        super().__init__()
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(images)
        lap = F.conv2d(gray, self.kernel, padding=1)
        return lap.abs()


class FocusPrior(nn.Module):
    """
    S = |ΔI|;  s_i = mean_{P_i} S;  B_ij = s_i s_j
    """

    def __init__(self, normalize: bool = False) -> None:
        super().__init__()
        self.laplacian = LaplacianSharpness()
        self.normalize = normalize

    def sharpness_map(self, images: torch.Tensor) -> torch.Tensor:
        s_map = self.laplacian(images)
        if self.normalize:
            # Per-image min-max to keep biases in a stable range
            flat = s_map.flatten(2)
            s_min = flat.min(dim=-1, keepdim=True).values.unsqueeze(-1)
            s_max = flat.max(dim=-1, keepdim=True).values.unsqueeze(-1)
            s_map = (s_map - s_min) / (s_max - s_min + 1e-6)
        return s_map

    def token_scores(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        return pool_map_to_tokens(
            self.sharpness_map(images), num_tokens, tokens_h, tokens_w
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
