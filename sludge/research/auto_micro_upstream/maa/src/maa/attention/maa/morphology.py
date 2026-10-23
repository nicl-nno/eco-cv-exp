"""Morphology-Ratio Attention prior / shape descriptor."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.attention.common import pool_map_to_tokens, rgb_to_gray, similarity_bias


class MorphologyPrior(nn.Module):
    """
    r_i = λ_max(C_i) / λ_min(C_i) from local gradient covariance;
    B_ij = exp(-(r_i - r_j)^2 / σ^2)
    """

    def __init__(self, sigma: float = 1.0, patch: int = 5, eps: float = 1e-4) -> None:
        super().__init__()
        self.sigma = sigma
        self.patch = patch
        self.eps = eps
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def ratio_map(self, images: torch.Tensor) -> torch.Tensor:
        gray = rgb_to_gray(images)
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        # Local second-moment estimates via average pooling
        k = self.patch
        gxx = F.avg_pool2d(gx * gx, k, stride=1, padding=k // 2)
        gyy = F.avg_pool2d(gy * gy, k, stride=1, padding=k // 2)
        gxy = F.avg_pool2d(gx * gy, k, stride=1, padding=k // 2)
        # Eigenvalues of [[gxx, gxy], [gxy, gyy]]
        trace = gxx + gyy
        det = gxx * gyy - gxy * gxy
        disc = torch.clamp(trace * trace - 4.0 * det, min=0.0)
        sqrt_disc = torch.sqrt(disc + 1e-8)
        lambda_max = 0.5 * (trace + sqrt_disc)
        lambda_min = 0.5 * (trace - sqrt_disc)
        ratio = (lambda_max + self.eps) / (lambda_min.abs() + self.eps)
        return ratio

    def patch_ratios(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute r_i = λ_max(C_i)/λ_min(C_i) on each token patch P_i.
        """
        import math

        gray = rgb_to_gray(images)
        b, _, h, w = gray.shape
        if tokens_h is None or tokens_w is None:
            side = int(math.sqrt(num_tokens))
            tokens_h = tokens_w = side if side * side == num_tokens else (1, num_tokens)[0]
            if side * side != num_tokens:
                tokens_h, tokens_w = 1, num_tokens
        # Unfold into patches approximating token regions
        patches = F.adaptive_avg_pool2d(gray, (tokens_h * self.patch, tokens_w * self.patch))
        patches = patches.unfold(2, self.patch, self.patch).unfold(3, self.patch, self.patch)
        # patches: B,1,Th,Tw,p,p
        patches = patches.reshape(b, tokens_h * tokens_w, self.patch, self.patch)
        # Gradients inside each patch
        ratios = []
        for i in range(patches.size(1)):
            p = patches[:, i : i + 1]  # B,1,p,p
            gx = F.conv2d(p, self.kx, padding=1)
            gy = F.conv2d(p, self.ky, padding=1)
            gxx = (gx * gx).mean(dim=(2, 3))
            gyy = (gy * gy).mean(dim=(2, 3))
            gxy = (gx * gy).mean(dim=(2, 3))
            trace = gxx + gyy
            det = gxx * gyy - gxy * gxy
            disc = torch.clamp(trace * trace - 4.0 * det, min=0.0)
            sqrt_disc = torch.sqrt(disc + 1e-8)
            lambda_max = 0.5 * (trace + sqrt_disc)
            lambda_min = 0.5 * (trace - sqrt_disc)
            ratios.append((lambda_max + self.eps) / (lambda_min.abs() + self.eps))
        return torch.cat(ratios, dim=1)

    def token_scores(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
        *,
        exact_patches: bool = False,
    ) -> torch.Tensor:
        if exact_patches:
            return self.patch_ratios(images, num_tokens, tokens_h, tokens_w)
        return pool_map_to_tokens(self.ratio_map(images), num_tokens, tokens_h, tokens_w)

    def forward(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> torch.Tensor:
        ratios = self.token_scores(images, num_tokens, tokens_h, tokens_w)
        return similarity_bias(ratios, sigma=self.sigma)
