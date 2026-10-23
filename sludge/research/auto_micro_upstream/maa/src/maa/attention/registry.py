"""Attention module registry: none / se / sa / cbam / maa (+ ablations)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from maa.attention.baselines.cbam import CBAMBlock
from maa.attention.baselines.none import NoAttention
from maa.attention.baselines.sa import SABlock
from maa.attention.baselines.se import SEBlock
from maa.attention.maa.block import MAAConfig, MicroscopyAwareAttention, build_maa_for_ablation
from maa.constants import ATTENTION_VARIANTS, MAA_PRIOR_NAMES


def build_feature_attention(name: str, channels: int, **kwargs: Any) -> nn.Module:
    """Build a spatial-feature attention adapter (SE/SA/CBAM/none)."""
    key = name.lower().strip()
    if key in ("none", "no", "identity", "-"):
        return NoAttention()
    if key == "se":
        return SEBlock(channels, reduction=int(kwargs.get("reduction", 16)))
    if key == "sa":
        return SABlock(kernel_size=int(kwargs.get("spatial_kernel", 7)))
    if key == "cbam":
        return CBAMBlock(
            channels,
            reduction=int(kwargs.get("reduction", 16)),
            spatial_kernel=int(kwargs.get("spatial_kernel", 7)),
        )
    raise ValueError(
        f"Feature attention {name!r} not supported; use one of {ATTENTION_VARIANTS} "
        "(MAA is token-level — use build_token_attention)."
    )


def build_token_attention(
    name: str,
    dim: int,
    *,
    disabled_prior: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Build token-level attention (MAA or ablated MAA)."""
    key = name.lower().strip()
    if key in ("maa", "ours", "proposed"):
        if disabled_prior is not None:
            if disabled_prior not in MAA_PRIOR_NAMES:
                raise ValueError(f"Unknown prior {disabled_prior}")
            return build_maa_for_ablation(dim, disabled_prior=disabled_prior, **kwargs)
        return MicroscopyAwareAttention(MAAConfig(dim=dim, **kwargs))
    if key.startswith("maa_w/o_") or key.startswith("maa_wo_"):
        prior = key.split("_", 2)[-1].replace("w/o_", "").replace("wo_", "")
        # names like maa_wo_boundary
        prior = key.replace("maa_w/o_", "").replace("maa_wo_", "")
        return build_maa_for_ablation(dim, disabled_prior=prior, **kwargs)
    raise ValueError(f"Token attention {name!r} not supported; expected 'maa' or ablation.")


def is_maa_variant(name: str) -> bool:
    key = name.lower().strip()
    return key in ("maa", "ours", "proposed") or key.startswith("maa_wo") or key.startswith("maa_w/o")
