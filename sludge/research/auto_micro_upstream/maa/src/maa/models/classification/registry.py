"""Registry for classification hosts."""

from __future__ import annotations

from typing import Callable, Optional

import torch.nn as nn

from maa.constants import ATTENTION_VARIANTS, CLASSIFICATION_BACKBONES, NUM_CLASSES
from maa.models.classification.base import ClassificationModelBase
from maa.models.classification.convnext_v2 import build_convnext_v2
from maa.models.classification.dinov3 import build_dinov3
from maa.models.classification.efficientnetv2 import build_efficientnetv2
from maa.models.classification.eva02 import build_eva02
from maa.models.classification.maxvit import build_maxvit
from maa.models.classification.swinv2 import build_swinv2

Builder = Callable[..., ClassificationModelBase]

_CLASSIFICATION_BUILDERS: dict[str, Builder] = {
    "efficientnetv2": build_efficientnetv2,
    "convnext_v2": build_convnext_v2,
    "swinv2": build_swinv2,
    "maxvit": build_maxvit,
    "eva02": build_eva02,
    "dinov3": build_dinov3,
}


def build_classification_model(
    backbone: str,
    attention: str,
    num_classes: int = NUM_CLASSES,
    *,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> ClassificationModelBase:
    """
    Factory for classification baselines.

    Parameters
    ----------
    backbone:
        One of ``CLASSIFICATION_BACKBONES`` (e.g. ``efficientnetv2``).
    attention:
        One of ``ATTENTION_VARIANTS``: ``none``, ``se``, ``sa``, ``cbam``, ``maa``.
    num_classes:
        Output logits width; defaults to ``NUM_CLASSES`` (5 microorganism classes).
    pretrained:
        Reserved for external checkpoint loading (default off).
    disabled_prior:
        MAA leave-one-out prior name for ablations.
    """
    key = backbone.lower().strip().replace("-", "_")
    if key not in _CLASSIFICATION_BUILDERS:
        raise ValueError(
            f"Unknown classification backbone {backbone!r}; "
            f"expected one of {CLASSIFICATION_BACKBONES}"
        )
    attn = attention.lower().strip()
    if attn not in ATTENTION_VARIANTS and not attn.startswith("maa"):
        raise ValueError(
            f"Unknown attention {attention!r}; expected one of {ATTENTION_VARIANTS}"
        )
    return _CLASSIFICATION_BUILDERS[key](
        attention=attn,
        num_classes=num_classes,
        pretrained=pretrained,
        disabled_prior=disabled_prior,
    )


def list_classification_backbones() -> tuple[str, ...]:
    return CLASSIFICATION_BACKBONES
