"""COCO reading, native 512 px crops and the article/fold-0 augmentations."""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

PATCH = 512
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_coco(root: Path) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    raw = json.loads((root / "annotations" / "instances.json").read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in raw["images"]}
    annotations: dict[int, list[dict]] = defaultdict(list)
    for row in raw["annotations"]:
        annotations[int(row["image_id"])].append(row)
    return images, annotations


def image_path(root: Path, info: dict) -> Path:
    path = root / "images" / Path(info["file_name"]).name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def make_mask(info: dict, annotations: list[dict]) -> np.ndarray:
    """Map COCO ids 1/3 to active/mature and id 2 to ignored pixels."""
    mask = Image.new("L", (int(info["width"]), int(info["height"])), 0)
    draw, ignored = ImageDraw.Draw(mask), []
    for annotation in annotations:
        category = int(annotation["category_id"])
        for polygon in annotation.get("segmentation", []):
            points = list(zip(polygon[::2], polygon[1::2]))
            if len(points) < 3:
                continue
            if category == 2:
                ignored.append(points)
            elif category == 1:
                draw.polygon(points, fill=1)
            elif category == 3:
                draw.polygon(points, fill=2)
    for points in ignored:
        draw.polygon(points, fill=IGNORE)
    return np.asarray(mask, dtype=np.uint8)


def normalize(image: np.ndarray) -> torch.Tensor:
    value = (image.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(value.transpose(2, 0, 1).copy()).float()


def crop_and_pad(image: np.ndarray, mask: np.ndarray, y: int, x: int) -> tuple[np.ndarray, np.ndarray]:
    image, mask = image[y:y + PATCH, x:x + PATCH], mask[y:y + PATCH, x:x + PATCH]
    dy, dx = PATCH - mask.shape[0], PATCH - mask.shape[1]
    if dy or dx:
        image = np.pad(image, ((0, dy), (0, dx), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, dy), (0, dx)), constant_values=IGNORE)
    return image, mask


def _shift_scale_rotate(image: np.ndarray, mask: np.ndarray,
                        scale_range: tuple[float, float] = (0.9, 1.1)) -> tuple[np.ndarray, np.ndarray]:
    if random.random() >= 0.5:
        return image, mask
    angle, scale = random.uniform(-45, 45), random.uniform(*scale_range)
    shift_x, shift_y = (random.uniform(-0.0625, 0.0625) * PATCH for _ in range(2))
    radians = math.radians(angle)
    c, s = math.cos(radians) / scale, math.sin(radians) / scale
    center = (PATCH - 1) / 2
    a, b, d, e = c, s, -s, c
    tx = center - a * (center + shift_x) - b * (center + shift_y)
    ty = center - d * (center + shift_x) - e * (center + shift_y)
    coefficients = (a, b, tx, d, e, ty)
    image = Image.fromarray(image).transform(
        (PATCH, PATCH), Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
    )
    mask = Image.fromarray(mask).transform(
        (PATCH, PATCH), Image.Transform.AFFINE, coefficients,
        resample=Image.Resampling.NEAREST, fillcolor=IGNORE,
    )
    return np.asarray(image), np.asarray(mask)


def _elastic(image: np.ndarray, mask: np.ndarray,
             probability: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    if random.random() >= probability:
        return image, mask
    dy = gaussian_filter(np.random.rand(PATCH, PATCH) * 2 - 1, 6, mode="reflect") * 40
    dx = gaussian_filter(np.random.rand(PATCH, PATCH) * 2 - 1, 6, mode="reflect") * 40
    yy, xx = np.meshgrid(np.arange(PATCH), np.arange(PATCH), indexing="ij")
    coordinates = (np.clip(yy + dy, 0, PATCH - 1), np.clip(xx + dx, 0, PATCH - 1))
    warped = np.stack([
        map_coordinates(image[..., channel], coordinates, order=1, mode="reflect")
        for channel in range(3)
    ], axis=-1)
    target = map_coordinates(mask, coordinates, order=0, mode="nearest")
    return warped.astype(np.uint8), target.astype(np.uint8)


def augment(image: np.ndarray, mask: np.ndarray,
            scale_range: tuple[float, float] = (0.9, 1.1),
            color_jitter_probability: float = 0.0,
            elastic_probability: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """Equivalent to the article pipeline; this is the implementation used by the best fold-0 run."""
    image, mask = _shift_scale_rotate(image, mask, scale_range)
    if random.random() < 0.5:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
    if random.random() < 0.5:
        image, mask = image[::-1].copy(), mask[::-1].copy()
    image, mask = _elastic(image, mask, elastic_probability)
    if random.random() < 0.5:
        image = np.asarray(ImageEnhance.Brightness(Image.fromarray(image)).enhance(random.uniform(0.8, 1.2)))
        image = np.asarray(ImageEnhance.Contrast(Image.fromarray(image)).enhance(random.uniform(0.8, 1.2)))
    if color_jitter_probability and random.random() < color_jitter_probability:
        image = np.asarray(ImageEnhance.Color(Image.fromarray(image)).enhance(random.uniform(0.7, 1.3)))
    return normalize(image), torch.from_numpy(mask.copy()).long()


class TrainPatches(Dataset):
    def __init__(self, root: Path, images: dict[int, dict], annotations: dict[int, list[dict]],
                 ids: list[int], patches_per_image: int = 8, foreground_probability: float = 0.7,
                 copy_paste_probability: float = 0.0, copy_paste_max_objects: int = 2,
                 scale_range: tuple[float, float] = (0.9, 1.1),
                 color_jitter_probability: float = 0.0,
                 elastic_probability: float = 0.3):
        self.root, self.images, self.annotations, self.ids = root, images, annotations, ids
        self.patches_per_image = patches_per_image
        self.foreground_probability = foreground_probability
        self.copy_paste_probability = copy_paste_probability
        self.copy_paste_max_objects = copy_paste_max_objects
        self.scale_range = tuple(map(float, scale_range))
        self.color_jitter_probability = float(color_jitter_probability)
        self.elastic_probability = float(elastic_probability)
        if not 0 <= copy_paste_probability <= 1 or copy_paste_max_objects < 1:
            raise ValueError("Invalid copy-paste probability or object count")
        if len(self.scale_range) != 2 or not 0 < self.scale_range[0] <= self.scale_range[1]:
            raise ValueError("Invalid augmentation scale range")
        if not 0 <= self.color_jitter_probability <= 1:
            raise ValueError("Invalid color jitter probability")
        if not 0 <= self.elastic_probability <= 1:
            raise ValueError("Invalid elastic probability")
        # Only objects from this fold's training images may become donors.
        self.donors = {category: [(image_id, annotation) for image_id in ids
                                  for annotation in annotations[image_id]
                                  if int(annotation["category_id"]) == category
                                  and annotation.get("segmentation")]
                       for category in (1, 3)}

    def _paste_one(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        available = [category for category, donors in self.donors.items() if donors]
        if not available:
            return image, mask
        donor_id, annotation = random.choice(self.donors[random.choice(available)])
        donor = np.asarray(Image.open(image_path(self.root, self.images[donor_id])).convert("RGB"))
        polygon_mask = Image.new("L", (donor.shape[1], donor.shape[0]), 0)
        drawer = ImageDraw.Draw(polygon_mask)
        for polygon in annotation["segmentation"]:
            points = list(zip(polygon[::2], polygon[1::2]))
            if len(points) >= 3:
                drawer.polygon(points, fill=255)
        box = polygon_mask.getbbox()
        if box is None:
            return image, mask
        cutout = Image.fromarray(donor).crop(box)
        silhouette = polygon_mask.crop(box)
        scale = random.uniform(0.8, 1.2)
        width, height = max(1, round(cutout.width * scale)), max(1, round(cutout.height * scale))
        if width >= PATCH or height >= PATCH:
            return image, mask
        cutout = cutout.resize((width, height), Image.Resampling.BILINEAR)
        silhouette = silhouette.resize((width, height), Image.Resampling.NEAREST)
        solid = np.asarray(silhouette) > 127
        if not solid.any():
            return image, mask
        for _ in range(20):
            x = random.randrange(PATCH - width + 1)
            y = random.randrange(PATCH - height + 1)
            region = mask[y:y + height, x:x + width]
            # Never overwrite a real object, an ignored pixel, or a previous paste.
            if np.all(region[solid] == 0):
                alpha = np.asarray(silhouette.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255
                # The soft edge is confined to the labelled silhouette.
                alpha *= solid
                original = image[y:y + height, x:x + width].astype(np.float32)
                pasted = np.asarray(cutout, dtype=np.float32)
                image[y:y + height, x:x + width] = np.clip(
                    original * (1 - alpha[..., None]) + pasted * alpha[..., None], 0, 255
                ).astype(np.uint8)
                region[solid] = 1 if int(annotation["category_id"]) == 1 else 2
                break
        return image, mask

    def __len__(self) -> int:
        return len(self.ids) * self.patches_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.ids[index % len(self.ids)]
        info = self.images[image_id]
        image = np.asarray(Image.open(image_path(self.root, info)).convert("RGB"))
        mask = make_mask(info, self.annotations[image_id])
        height, width = mask.shape
        foreground = np.argwhere((mask == 1) | (mask == 2))
        if len(foreground) and random.random() < self.foreground_probability:
            center_y, center_x = foreground[random.randrange(len(foreground))]
            y = int(np.clip(center_y - random.randrange(PATCH), 0, max(0, height - PATCH)))
            x = int(np.clip(center_x - random.randrange(PATCH), 0, max(0, width - PATCH)))
        else:
            y = random.randrange(max(1, height - PATCH + 1))
            x = random.randrange(max(1, width - PATCH + 1))
        image, mask = crop_and_pad(image, mask, y, x)
        if self.copy_paste_probability and random.random() < self.copy_paste_probability:
            image, mask = image.copy(), mask.copy()
            for _ in range(random.randint(1, self.copy_paste_max_objects)):
                image, mask = self._paste_one(image, mask)
        return augment(image, mask, self.scale_range,
                       self.color_jitter_probability, self.elastic_probability)
