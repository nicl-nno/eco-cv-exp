"""Registry for segmentation hosts."""

from __future__ import annotations

from typing import Callable, Optional

from maa.constants import ATTENTION_VARIANTS, NUM_CLASSES, SEGMENTATION_BACKBONES
from maa.models.segmentation.base import SegmentationModelBase
from maa.models.segmentation.mask2former import build_mask2former
from maa.models.segmentation.mask_dino import build_mask_dino
from maa.models.segmentation.oneformer import build_oneformer
from maa.models.segmentation.segman import build_segman
from maa.models.segmentation.segnext import build_segnext

Builder = Callable[..., SegmentationModelBase]

_SEGMENTATION_BUILDERS: dict[str, Builder] = {
    "mask_dino": build_mask_dino,
    "mask2former": build_mask2former,
    "oneformer": build_oneformer,
    "segnext": build_segnext,
    "segman": build_segman,
}


def build_segmentation_model(
    backbone: str,
    attention: str,
    num_classes: int = NUM_CLASSES,
    *,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> SegmentationModelBase:
    """
    Factory for segmentation baselines.

    Parameters
    ----------
    backbone:
        One of ``SEGMENTATION_BACKBONES`` (e.g. ``mask2former``).
    attention:
        One of ``ATTENTION_VARIANTS`` injected at decoder fusion.
    num_classes:
        Per-pixel class count (default ``NUM_CLASSES``).
    pretrained:
        Reserved for future weight loading hooks.
    disabled_prior:
        MAA leave-one-out prior for ablations.
    """
    key = backbone.lower().strip().replace("-", "_")
    if key not in _SEGMENTATION_BUILDERS:
        raise ValueError(
            f"Unknown segmentation backbone {backbone!r}; expected one of {SEGMENTATION_BACKBONES}"
        )
    attn = attention.lower().strip()
    if attn not in ATTENTION_VARIANTS and not attn.startswith("maa"):
        raise ValueError(
            f"Unknown attention {attention!r}; expected one of {ATTENTION_VARIANTS}"
        )
    _ = pretrained
    return _SEGMENTATION_BUILDERS[key](
        attention=attn,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )


def list_segmentation_backbones() -> tuple[str, ...]:
    return SEGMENTATION_BACKBONES
