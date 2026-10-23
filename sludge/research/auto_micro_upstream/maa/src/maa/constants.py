"""Constants for the MAA microscopy benchmark."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "background",
    "micrococci",
    "diplococci",
    "streptococci",
    "bacilli",
)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_IDX.items()}

NUM_CLASSES: int = len(CLASS_NAMES)

DETECTION_CLASS_NAMES: tuple[str, ...] = (
    "micrococci",
    "diplococci",
    "streptococci",
    "bacilli",
    "background",
)

CLASSIFICATION_PROMPTS: tuple[str, ...] = (
    "a microscopy image of a background",
    "a microscopy image of a micrococci",
    "a microscopy image of a diplococci",
    "a microscopy image of a streptococci",
    "a microscopy image of a bacilli",
)

GROUNDING_DINO_PROMPT: str = "background. micrococci. diplococci. streptococci. bacilli."

EXPERIMENT_SEEDS: tuple[int, ...] = (42, 123, 321)
INPUT_SIZE: int = 512
DEFAULT_EPOCHS: int = 50
OPTIMIZER_NAME: str = "AdamW"
LR_SCHEDULE: str = "cosine"

ATTENTION_VARIANTS: tuple[str, ...] = (
    "none",
    "se",
    "sa",
    "cbam",
    "maa",
)

CLASSIFICATION_BACKBONES: tuple[str, ...] = (
    "efficientnetv2",
    "convnext_v2",
    "swinv2",
    "maxvit",
    "eva02",
    "dinov3",
)

DETECTION_BACKBONES: tuple[str, ...] = (
    "yolov13",
    "rt_detr_v4",
    "dino_detr",
    "grounding_dino",
    "co_detr",
    "internimage_h",
)

SEGMENTATION_BACKBONES: tuple[str, ...] = (
    "mask_dino",
    "mask2former",
    "oneformer",
    "segnext",
    "segman",
)

MAA_PRIOR_NAMES: tuple[str, ...] = (
    "boundary",
    "focus",
    "organelle",
    "compactness",
    "morphology",
    "density",
)
