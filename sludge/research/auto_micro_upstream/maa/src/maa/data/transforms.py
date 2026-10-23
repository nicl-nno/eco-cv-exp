"""Train/validation transforms for classification, detection, and segmentation.

Augmentations: horizontal/vertical flip, rotation, scale, brightness/contrast
jitter, Gaussian noise, and Gaussian blur. Detection and segmentation apply the
same geometric transforms jointly to images and annotations (boxes / masks).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

from maa.constants import INPUT_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformConfig:
    input_size: int = INPUT_SIZE
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD
    # Augmentation strengths
    rotation_degrees: float = 15.0
    scale_range: tuple[float, float] = (0.85, 1.15)
    brightness: float = 0.2
    contrast: float = 0.2
    gaussian_noise_std: float = 0.02
    gaussian_blur_sigma: tuple[float, float] = (0.1, 1.0)
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate_prob: float = 0.5
    scale_prob: float = 0.5
    color_jitter_prob: float = 0.5
    noise_prob: float = 0.3
    blur_prob: float = 0.2


def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _to_mask_pil(mask: Any) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask
    if isinstance(mask, np.ndarray):
        return Image.fromarray(mask.astype(np.uint8))
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
        return Image.fromarray(arr.astype(np.uint8))
    raise TypeError(f"Unsupported mask type: {type(mask)!r}")


def _normalize_tensor(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean_t) / std_t


def _resize_square(image: Image.Image, size: int, *, resample: int = Image.BILINEAR) -> Image.Image:
    return TF.resize(image, [size, size], interpolation=resample)


def _apply_photometric_pil(
    image: Image.Image,
    cfg: TransformConfig,
    rng: random.Random,
) -> Image.Image:
    if rng.random() < cfg.color_jitter_prob:
        brightness_factor = 1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
        contrast_factor = 1.0 + rng.uniform(-cfg.contrast, cfg.contrast)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)
    if rng.random() < cfg.noise_prob:
        arr = np.asarray(image).astype(np.float32) / 255.0
        noise = rng.gauss(0.0, cfg.gaussian_noise_std)
        arr = np.clip(arr + np.random.normal(0.0, noise, arr.shape), 0.0, 1.0)
        image = Image.fromarray((arr * 255.0).astype(np.uint8))
    if rng.random() < cfg.blur_prob:
        sigma = rng.uniform(*cfg.gaussian_blur_sigma)
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return image


@dataclass
class GeometricParams:
    hflip: bool = False
    vflip: bool = False
    angle: float = 0.0
    scale: float = 1.0


def _sample_geometric(cfg: TransformConfig, rng: random.Random) -> GeometricParams:
    params = GeometricParams()
    if rng.random() < cfg.hflip_prob:
        params.hflip = True
    if rng.random() < cfg.vflip_prob:
        params.vflip = True
    if rng.random() < cfg.rotate_prob:
        params.angle = rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
    if rng.random() < cfg.scale_prob:
        params.scale = rng.uniform(*cfg.scale_range)
    return params


def _transform_boxes_xyxy(
    boxes: torch.Tensor,
    orig_w: int,
    orig_h: int,
    params: GeometricParams,
    out_size: int,
) -> torch.Tensor:
    """Transform axis-aligned boxes in absolute xyxy coordinates."""
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    boxes = boxes.clone().float()
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    if params.hflip:
        cx = orig_w - cx
    if params.vflip:
        cy = orig_h - cy

    if params.angle != 0.0 or params.scale != 1.0:
        rad = math.radians(params.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        nx = orig_w / 2.0
        ny = orig_h / 2.0
        x = (cx - nx) * params.scale
        y = (cy - ny) * params.scale
        cx = x * cos_a - y * sin_a + nx
        cy = x * sin_a + y * cos_a + ny
        w = w * params.scale
        h = h * params.scale

    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0

    sx = out_size / float(orig_w)
    sy = out_size / float(orig_h)
    x1 *= sx
    x2 *= sx
    y1 *= sy
    y2 *= sy

    out = torch.stack([x1, y1, x2, y2], dim=1)
    out[:, 0::2] = out[:, 0::2].clamp(0, out_size)
    out[:, 1::2] = out[:, 1::2].clamp(0, out_size)
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out[keep]


def _apply_geometric_pil(
    image: Image.Image,
    params: GeometricParams,
    out_size: int,
    *,
    is_mask: bool = False,
) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    w, h = image.size

    if params.hflip:
        image = TF.hflip(image)
    if params.vflip:
        image = TF.vflip(image)
    if params.scale != 1.0:
        nw = max(1, int(round(w * params.scale)))
        nh = max(1, int(round(h * params.scale)))
        image = TF.resize(image, [nh, nw], interpolation=resample)
        w, h = image.size
    if params.angle != 0.0:
        image = TF.rotate(image, params.angle, interpolation=resample, fill=0)
    image = _resize_square(image, out_size, resample=resample)
    return image


class ClassificationTrainTransform:
    """512² training pipeline for image-level classification."""

    def __init__(self, cfg: Optional[TransformConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or TransformConfig()
        self._rng = random.Random(seed)

    def __call__(self, image: Any) -> torch.Tensor:
        pil = _to_pil(image)
        params = _sample_geometric(self.cfg, self._rng)
        pil = _apply_geometric_pil(pil, params, self.cfg.input_size, is_mask=False)
        pil = _apply_photometric_pil(pil, self.cfg, self._rng)
        tensor = TF.to_tensor(pil)
        return _normalize_tensor(tensor, self.cfg.mean, self.cfg.std)


class ClassificationValTransform:
    def __init__(self, cfg: Optional[TransformConfig] = None) -> None:
        self.cfg = cfg or TransformConfig()

    def __call__(self, image: Any) -> torch.Tensor:
        pil = _resize_square(_to_pil(image), self.cfg.input_size)
        tensor = TF.to_tensor(pil)
        return _normalize_tensor(tensor, self.cfg.mean, self.cfg.std)


class DetectionTrainTransform:
    """Joint image + box transform for detection (COCO xyxy absolute coords)."""

    def __init__(self, cfg: Optional[TransformConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or TransformConfig()
        self._rng = random.Random(seed)

    def __call__(
        self,
        image: Any,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pil = _to_pil(image)
        orig_w, orig_h = pil.size
        params = _sample_geometric(self.cfg, self._rng)
        pil = _apply_geometric_pil(pil, params, self.cfg.input_size, is_mask=False)
        pil = _apply_photometric_pil(pil, self.cfg, self._rng)
        boxes = _transform_boxes_xyxy(boxes, orig_w, orig_h, params, self.cfg.input_size)
        if boxes.numel() == 0:
            labels = labels.new_zeros((0,), dtype=labels.dtype)
        else:
            labels = labels[: boxes.shape[0]]
        tensor = _normalize_tensor(TF.to_tensor(pil), self.cfg.mean, self.cfg.std)
        return tensor, boxes, labels


class DetectionValTransform:
    def __init__(self, cfg: Optional[TransformConfig] = None) -> None:
        self.cfg = cfg or TransformConfig()

    def __call__(
        self,
        image: Any,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pil = _to_pil(image)
        orig_w, orig_h = pil.size
        pil = _resize_square(pil, self.cfg.input_size)
        params = GeometricParams()
        boxes = _transform_boxes_xyxy(boxes, orig_w, orig_h, params, self.cfg.input_size)
        if boxes.numel() == 0:
            labels = labels.new_zeros((0,), dtype=labels.dtype)
        else:
            labels = labels[: boxes.shape[0]]
        tensor = _normalize_tensor(TF.to_tensor(pil), self.cfg.mean, self.cfg.std)
        return tensor, boxes, labels


class SegmentationTrainTransform:
    """Joint image + mask transform for semantic segmentation."""

    def __init__(self, cfg: Optional[TransformConfig] = None, seed: Optional[int] = None) -> None:
        self.cfg = cfg or TransformConfig()
        self._rng = random.Random(seed)

    def __call__(self, image: Any, mask: Any) -> tuple[torch.Tensor, torch.Tensor]:
        pil = _to_pil(image)
        mask_pil = _to_mask_pil(mask)
        params = _sample_geometric(self.cfg, self._rng)
        pil = _apply_geometric_pil(pil, params, self.cfg.input_size, is_mask=False)
        mask_pil = _apply_geometric_pil(mask_pil, params, self.cfg.input_size, is_mask=True)
        pil = _apply_photometric_pil(pil, self.cfg, self._rng)
        image_t = _normalize_tensor(TF.to_tensor(pil), self.cfg.mean, self.cfg.std)
        mask_t = torch.from_numpy(np.asarray(mask_pil, dtype=np.int64))
        if mask_t.ndim == 3:
            mask_t = mask_t[0]
        return image_t, mask_t


class SegmentationValTransform:
    def __init__(self, cfg: Optional[TransformConfig] = None) -> None:
        self.cfg = cfg or TransformConfig()

    def __call__(self, image: Any, mask: Any) -> tuple[torch.Tensor, torch.Tensor]:
        pil = _resize_square(_to_pil(image), self.cfg.input_size)
        mask_pil = _resize_square(_to_mask_pil(mask), self.cfg.input_size, resample=Image.NEAREST)
        image_t = _normalize_tensor(TF.to_tensor(pil), self.cfg.mean, self.cfg.std)
        mask_t = torch.from_numpy(np.asarray(mask_pil, dtype=np.int64))
        if mask_t.ndim == 3:
            mask_t = mask_t[0]
        return image_t, mask_t


def build_classification_transforms(
    train: bool,
    cfg: Optional[TransformConfig] = None,
    seed: Optional[int] = None,
) -> Callable[..., Any]:
    if train:
        return ClassificationTrainTransform(cfg, seed=seed)
    return ClassificationValTransform(cfg)


def build_detection_transforms(
    train: bool,
    cfg: Optional[TransformConfig] = None,
    seed: Optional[int] = None,
) -> DetectionTrainTransform | DetectionValTransform:
    if train:
        return DetectionTrainTransform(cfg, seed=seed)
    return DetectionValTransform(cfg)


def build_segmentation_transforms(
    train: bool,
    cfg: Optional[TransformConfig] = None,
    seed: Optional[int] = None,
) -> SegmentationTrainTransform | SegmentationValTransform:
    if train:
        return SegmentationTrainTransform(cfg, seed=seed)
    return SegmentationValTransform(cfg)
