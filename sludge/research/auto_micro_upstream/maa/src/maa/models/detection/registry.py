"""Registry for detection hosts."""

from __future__ import annotations

from typing import Callable, Optional

from maa.constants import ATTENTION_VARIANTS, DETECTION_BACKBONES, DETECTION_CLASS_NAMES
from maa.models.detection.base import DetectionModelBase
from maa.models.detection.co_detr import build_co_detr
from maa.models.detection.dino_detr import build_dino_detr
from maa.models.detection.grounding_dino import build_grounding_dino
from maa.models.detection.internimage_h import build_internimage_h
from maa.models.detection.rt_detr_v4 import build_rt_detr_v4
from maa.models.detection.yolov13 import build_yolov13

Builder = Callable[..., DetectionModelBase]

_DETECTION_BUILDERS: dict[str, Builder] = {
    "yolov13": build_yolov13,
    "rt_detr_v4": build_rt_detr_v4,
    "dino_detr": build_dino_detr,
    "grounding_dino": build_grounding_dino,
    "co_detr": build_co_detr,
    "internimage_h": build_internimage_h,
}


def build_detection_model(
    backbone: str,
    attention: str,
    num_classes: int = len(DETECTION_CLASS_NAMES),
    *,
    pretrained: bool = False,
    disabled_prior: Optional[str] = None,
) -> DetectionModelBase:
    """
    Factory for detection baselines.

    Parameters
    ----------
    backbone:
        One of ``DETECTION_BACKBONES`` (e.g. ``yolov13``, ``grounding_dino``).
    attention:
        One of ``ATTENTION_VARIANTS`` injected at FPN fusion.
    num_classes:
        Detection categories (default: ``DETECTION_CLASS_NAMES`` length).
    pretrained:
        Reserved for future weight loading hooks.
    disabled_prior:
        MAA leave-one-out prior for ablations.
    """
    key = backbone.lower().strip().replace("-", "_")
    if key not in _DETECTION_BUILDERS:
        raise ValueError(
            f"Unknown detection backbone {backbone!r}; expected one of {DETECTION_BACKBONES}"
        )
    attn = attention.lower().strip()
    if attn not in ATTENTION_VARIANTS and not attn.startswith("maa"):
        raise ValueError(
            f"Unknown attention {attention!r}; expected one of {ATTENTION_VARIANTS}"
        )
    _ = pretrained
    return _DETECTION_BUILDERS[key](
        attention=attn,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )


def list_detection_backbones() -> tuple[str, ...]:
    return DETECTION_BACKBONES
