"""MAA model factories for classification, detection, and segmentation."""

from maa.models.adapters import FeatureMapAttentionAdapter, TokenAttentionAdapter, build_attention_adapter
from maa.models.classification import (
    ClassificationModelBase,
    build_classification_model,
    list_classification_backbones,
)
from maa.models.detection import (
    DetectionModelBase,
    DetectionOutput,
    build_detection_model,
    list_detection_backbones,
)
from maa.models.segmentation import (
    SegmentationModelBase,
    build_segmentation_model,
    list_segmentation_backbones,
)

__all__ = [
    "FeatureMapAttentionAdapter",
    "TokenAttentionAdapter",
    "build_attention_adapter",
    "ClassificationModelBase",
    "build_classification_model",
    "list_classification_backbones",
    "DetectionModelBase",
    "DetectionOutput",
    "build_detection_model",
    "list_detection_backbones",
    "SegmentationModelBase",
    "build_segmentation_model",
    "list_segmentation_backbones",
]
