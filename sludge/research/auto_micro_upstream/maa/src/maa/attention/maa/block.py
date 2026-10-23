"""Microscopy-Aware Non-Local Attention Block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from maa.attention.common import SoftmaxAttention
from maa.attention.maa.boundary import BoundaryPrior
from maa.attention.maa.compactness import CompactnessPrior
from maa.attention.maa.density import DensityPrior
from maa.attention.maa.focus import FocusPrior
from maa.attention.maa.morphology import MorphologyPrior
from maa.attention.maa.organelle import OrganellePrior
from maa.constants import MAA_PRIOR_NAMES


@dataclass
class MAAConfig:
    dim: int = 256
    num_heads: int = 8
    dropout: float = 0.0
    compactness_sigma: float = 0.35
    morphology_sigma: float = 1.0
    density_sigma: float = 0.35
    use_foreground_compactness: bool = True
    learnable_edge: bool = True
    enable_boundary: bool = True
    enable_focus: bool = True
    enable_organelle: bool = True
    enable_compactness: bool = True
    enable_morphology: bool = True
    enable_density: bool = True


class MicroscopyAwareAttention(nn.Module):
    """
    A* = softmax( QK^T / sqrt(d_k) + Σ_m λ_m B^(m) )
    Y  = A* V

    Six biologically motivated priors with learnable λ_m scalars.
    """

    def __init__(self, config: MAAConfig | None = None, **kwargs) -> None:
        super().__init__()
        cfg = config or MAAConfig(**kwargs)
        self.config = cfg
        self.attn = SoftmaxAttention(cfg.dim, cfg.num_heads, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim * 4, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

        self.boundary = BoundaryPrior(learnable_edge=cfg.learnable_edge)
        self.focus = FocusPrior()
        self.organelle = OrganellePrior()
        self.compactness = CompactnessPrior(
            sigma=cfg.compactness_sigma,
            use_foreground=cfg.use_foreground_compactness,
        )
        self.morphology = MorphologyPrior(sigma=cfg.morphology_sigma)
        self.density = DensityPrior(sigma=cfg.density_sigma)

        # Learnable λ_m for each prior
        self.lambda_boundary = nn.Parameter(torch.tensor(0.1))
        self.lambda_focus = nn.Parameter(torch.tensor(0.1))
        self.lambda_organelle = nn.Parameter(torch.tensor(0.1))
        self.lambda_compactness = nn.Parameter(torch.tensor(0.1))
        self.lambda_morphology = nn.Parameter(torch.tensor(0.1))
        self.lambda_density = nn.Parameter(torch.tensor(0.1))

        self.enable = {
            "boundary": cfg.enable_boundary,
            "focus": cfg.enable_focus,
            "organelle": cfg.enable_organelle,
            "compactness": cfg.enable_compactness,
            "morphology": cfg.enable_morphology,
            "density": cfg.enable_density,
        }

    def set_prior_enabled(self, name: str, enabled: bool) -> None:
        if name not in self.enable:
            raise KeyError(f"Unknown prior {name}; expected one of {MAA_PRIOR_NAMES}")
        self.enable[name] = enabled

    def disable_prior(self, name: str) -> None:
        """Leave-one-out ablation helper."""
        self.set_prior_enabled(name, False)

    def compute_bias(
        self,
        images: torch.Tensor,
        num_tokens: int,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts: dict[str, torch.Tensor] = {}
        bias = None

        def accumulate(name: str, lam: nn.Parameter, module: nn.Module) -> None:
            nonlocal bias
            if not self.enable[name]:
                return
            b_m = module(images, num_tokens, tokens_h, tokens_w)
            parts[name] = b_m
            term = lam * b_m
            bias = term if bias is None else bias + term

        accumulate("boundary", self.lambda_boundary, self.boundary)
        accumulate("focus", self.lambda_focus, self.focus)
        accumulate("organelle", self.lambda_organelle, self.organelle)
        accumulate("compactness", self.lambda_compactness, self.compactness)
        accumulate("morphology", self.lambda_morphology, self.morphology)
        accumulate("density", self.lambda_density, self.density)

        if bias is None:
            bias = torch.zeros(
                images.size(0),
                num_tokens,
                num_tokens,
                device=images.device,
                dtype=images.dtype,
            )
        return bias, parts

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        tokens_h: Optional[int] = None,
        tokens_w: Optional[int] = None,
        return_attn: bool = False,
    ):
        """
        Args:
            tokens: (B, N, C) visual tokens
            images: (B, 3, H, W) original (or resized) microscopy crop for priors
        """
        b, n, c = tokens.shape
        if c != self.config.dim:
            raise ValueError(f"token dim {c} != MAA dim {self.config.dim}")
        bias, parts = self.compute_bias(images, n, tokens_h, tokens_w)
        x = self.norm1(tokens)
        out, attn = self.attn(x, bias=bias)
        tokens = tokens + out
        tokens = tokens + self.mlp(self.norm2(tokens))
        if return_attn:
            return tokens, {"attn": attn, "bias_parts": parts, "bias": bias}
        return tokens


class MicroscopyAwareAttentionStack(nn.Module):
    """Stack of MAA blocks for deeper token refinement."""

    def __init__(self, depth: int = 2, config: MAAConfig | None = None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MicroscopyAwareAttention(config=config) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor, images: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens, images, **kwargs)
        return tokens


def build_maa_for_ablation(dim: int, disabled_prior: str | None = None, **kwargs) -> MicroscopyAwareAttention:
    """Factory for full MAA or leave-one-out prior variants."""
    cfg_kwargs = dict(kwargs)
    cfg_kwargs.setdefault("dim", dim)
    if disabled_prior is not None:
        key = f"enable_{disabled_prior}"
        if key not in {
            "enable_boundary",
            "enable_focus",
            "enable_organelle",
            "enable_compactness",
            "enable_morphology",
            "enable_density",
        }:
            raise ValueError(f"Cannot disable unknown prior {disabled_prior}")
        cfg_kwargs[key] = False
    return MicroscopyAwareAttention(MAAConfig(**cfg_kwargs))
