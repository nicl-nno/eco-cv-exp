"""Semantic segmentation dataset: paired images and pixel masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from maa.constants import CLASS_NAMES, NUM_CLASSES
from maa.data.splits import load_split_manifest, split_by_ids


@dataclass(frozen=True)
class SegmentationRecord:
    stem: str
    image_path: Path
    mask_path: Path


def _discover_pairs(
    image_dir: Path,
    mask_dir: Path,
    *,
    image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    mask_suffixes: tuple[str, ...] = (".png", ".tif", ".tiff"),
) -> list[SegmentationRecord]:
    records: list[SegmentationRecord] = []
    for img_path in sorted(image_dir.rglob("*")):
        if img_path.suffix.lower() not in image_suffixes:
            continue
        stem = img_path.stem
        mask_path = None
        for suf in mask_suffixes:
            candidate = mask_dir / f"{stem}{suf}"
            if candidate.exists():
                mask_path = candidate
                break
            candidate = mask_dir / img_path.relative_to(image_dir).with_suffix(suf)
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            continue
        records.append(SegmentationRecord(stem=stem, image_path=img_path, mask_path=mask_path))
    if not records:
        raise FileNotFoundError(f"No image/mask pairs under {image_dir} and {mask_dir}")
    return records


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int64)


class MicroscopySegmentationDataset(Dataset):
    """
    Paired microscopy image + semantic mask dataset.

    Directory layout::

        images/<name>.png
        masks/<name>.png

    Mask values are integer class indices aligned with ``maa.constants.CLASS_NAMES``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        image_subdir: str = "images",
        mask_subdir: str = "masks",
        split: str | None = None,
        split_manifest: str | Path | None = None,
        seed: int = 42,
        transform: Callable[[Any, Any], tuple[torch.Tensor, torch.Tensor]] | None = None,
        ignore_index: int = 255,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.ignore_index = ignore_index
        self.seed = seed

        image_dir = self.root / image_subdir
        mask_dir = self.root / mask_subdir
        records = _discover_pairs(image_dir, mask_dir)

        if split_manifest is not None:
            split_map = load_split_manifest(split_manifest)
            allowed = set(split_map[split or "train"])
            records = [r for r in records if r.stem in allowed or str(r.image_path) in allowed]
        elif split is not None:
            stems = [r.stem for r in records]
            buckets = split_by_ids(stems, seed=seed)
            allowed = set(buckets[split])
            records = [r for r in records if r.stem in allowed]

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        with Image.open(rec.image_path) as img:
            image = img.convert("RGB")
        mask = _read_mask(rec.mask_path)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            mask = torch.from_numpy(mask)

        return {
            "image": image,
            "mask": mask.long(),
            "stem": rec.stem,
            "path": str(rec.image_path),
        }

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    @property
    def class_names(self) -> tuple[str, ...]:
        return CLASS_NAMES
