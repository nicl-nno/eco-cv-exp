"""Zenodo-style microscopy dataset layout and loaders.

Expected layout after unpacking class archives::

    root/
      monococci/images/*.png|jpg
      monococci/masks/*.png
      diplococci/...
      bacilli/...
      streptococci/...
      backgrounds/...   # or background/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image

from maa.constants import CLASS_NAMES, CLASS_TO_IDX, DETECTION_CLASS_NAMES

logger = logging.getLogger(__name__)

CLASS_DIR_ALIASES: dict[str, str] = {
    "background": "background",
    "backgrounds": "background",
    "monococci": "micrococci",
    "micrococci": "micrococci",
    "diplococci": "diplococci",
    "streptococci": "streptococci",
    "bacilli": "bacilli",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def canonicalize_class_name(name: str) -> str | None:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return CLASS_DIR_ALIASES.get(key)


def discover_class_roots(root: str | Path) -> dict[str, Path]:
    """Map canonical class name -> directory containing images/ and masks/."""
    root = Path(root)
    found: dict[str, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        canon = canonicalize_class_name(child.name)
        if canon is None:
            # Flat layout: root/images + root/masks with class in filename prefix
            continue
        images = child / "images"
        masks = child / "masks"
        if images.is_dir() and masks.is_dir():
            found[canon] = child
            continue
        # Some archives put files directly under the class folder
        if any(p.suffix.lower() in IMAGE_SUFFIXES for p in child.iterdir() if p.is_file()):
            found[canon] = child
    return found


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "mask" not in p.parts
    ]
    # Prefer images/ subdirectory contents when present
    under_images = [p for p in files if "images" in p.parts]
    return sorted(under_images or files)


def _mask_for_image(image_path: Path, class_root: Path) -> Path | None:
    stem = image_path.stem
    candidates = [
        class_root / "masks" / f"{stem}.png",
        class_root / "masks" / f"{stem}.PNG",
        class_root / "masks" / image_path.name,
        class_root / f"{stem}_mask.png",
        class_root / "mask" / f"{stem}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fuzzy: same stem anywhere under masks/
    masks_dir = class_root / "masks"
    if masks_dir.is_dir():
        matches = list(masks_dir.rglob(f"{stem}.*"))
        if matches:
            return matches[0]
    return None


@dataclass
class SampleRecord:
    image_path: str
    mask_path: str | None
    class_name: str
    class_id: int
    split: str = "unassigned"
    bbox: list[float] | None = None  # xyxy absolute pixels
    width: int | None = None
    height: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "split": self.split,
            "bbox": self.bbox,
            "width": self.width,
            "height": self.height,
            "meta": self.meta,
        }


def mask_to_bbox(mask: np.ndarray, min_area: int = 1) -> list[float] | None:
    """Binary mask -> xyxy bbox; None if empty."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    fg = mask > 0
    if not np.any(fg):
        return None
    ys, xs = np.where(fg)
    if ys.size < min_area:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, x1 + 1.0, y1 + 1.0]


def foreground_fraction(mask: np.ndarray) -> float:
    if mask.ndim == 3:
        mask = mask[..., 0]
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum()) / float(total)


def scan_zenodo_dataset(
    root: str | Path,
    *,
    compute_bbox: bool = True,
    compute_size: bool = True,
    limit_per_class: int | None = None,
) -> list[SampleRecord]:
    """Walk a multi-class Zenodo unpack and build sample records."""
    root = Path(root).resolve()
    class_roots = discover_class_roots(root)
    if not class_roots:
        # Flat images/ + masks/ with class subfolders or prefixes
        images_dir = root / "images"
        masks_dir = root / "masks"
        if images_dir.is_dir():
            return _scan_flat(images_dir, masks_dir if masks_dir.is_dir() else None, compute_bbox, compute_size, limit_per_class)
        raise FileNotFoundError(
            f"No class folders found under {root}. Expected monococci/, diplococci/, ..."
        )

    records: list[SampleRecord] = []
    for class_name, class_root in class_roots.items():
        images_dir = class_root / "images"
        image_paths = _list_images(images_dir if images_dir.is_dir() else class_root)
        if limit_per_class is not None:
            image_paths = image_paths[:limit_per_class]
        class_id = CLASS_TO_IDX.get(class_name, CLASS_TO_IDX.get("background", 0))
        for img_path in image_paths:
            mask_path = _mask_for_image(img_path, class_root)
            bbox = None
            width = height = None
            if compute_size or compute_bbox:
                with Image.open(img_path) as im:
                    width, height = im.size
            if compute_bbox and mask_path is not None:
                with Image.open(mask_path) as mm:
                    mask_arr = np.array(mm)
                bbox = mask_to_bbox(mask_arr)
            records.append(
                SampleRecord(
                    image_path=str(img_path.relative_to(root)).replace("\\", "/"),
                    mask_path=(
                        str(mask_path.relative_to(root)).replace("\\", "/")
                        if mask_path is not None
                        else None
                    ),
                    class_name=class_name,
                    class_id=class_id,
                    bbox=bbox,
                    width=width,
                    height=height,
                )
            )
    logger.info("Scanned %d samples across %d classes from %s", len(records), len(class_roots), root)
    return records


def _scan_flat(
    images_dir: Path,
    masks_dir: Path | None,
    compute_bbox: bool,
    compute_size: bool,
    limit_per_class: int | None,
) -> list[SampleRecord]:
    root = images_dir.parent
    records: list[SampleRecord] = []
    per_class_count: dict[str, int] = {}
    for img_path in sorted(images_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        # Infer class from parent folder or filename prefix
        parent = img_path.parent.name.lower()
        canon = canonicalize_class_name(parent)
        if canon is None:
            prefix = img_path.stem.split("_")[0]
            canon = canonicalize_class_name(prefix) or "background"
        if limit_per_class is not None and per_class_count.get(canon, 0) >= limit_per_class:
            continue
        per_class_count[canon] = per_class_count.get(canon, 0) + 1
        mask_path = None
        if masks_dir is not None:
            cand = masks_dir / img_path.relative_to(images_dir)
            if cand.exists():
                mask_path = cand
            else:
                alt = masks_dir / f"{img_path.stem}.png"
                if alt.exists():
                    mask_path = alt
        bbox = width = height = None
        if compute_size:
            with Image.open(img_path) as im:
                width, height = im.size
        if compute_bbox and mask_path is not None:
            with Image.open(mask_path) as mm:
                bbox = mask_to_bbox(np.array(mm))
        records.append(
            SampleRecord(
                image_path=str(img_path.relative_to(root)).replace("\\", "/"),
                mask_path=(
                    str(mask_path.relative_to(root)).replace("\\", "/") if mask_path else None
                ),
                class_name=canon,
                class_id=CLASS_TO_IDX.get(canon, 0),
                bbox=bbox,
                width=width,
                height=height,
            )
        )
    return records


def write_manifest(records: Sequence[SampleRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "num_samples": len(records),
        "classes": list(CLASS_NAMES),
        "detection_classes": list(DETECTION_CLASS_NAMES),
        "samples": [r.to_dict() for r in records],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote manifest %s (%d samples)", path, len(records))


def load_manifest(path: str | Path) -> list[SampleRecord]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = raw["samples"] if isinstance(raw, dict) else raw
    out: list[SampleRecord] = []
    for s in samples:
        out.append(
            SampleRecord(
                image_path=s["image_path"],
                mask_path=s.get("mask_path"),
                class_name=s["class_name"],
                class_id=int(s["class_id"]),
                split=str(s.get("split", "unassigned")),
                bbox=s.get("bbox"),
                width=s.get("width"),
                height=s.get("height"),
                meta=dict(s.get("meta", {})),
            )
        )
    return out


def export_coco_detection(
    records: Sequence[SampleRecord],
    path: str | Path,
    *,
    split: str | None = None,
) -> None:
    """Export COCO-style instances JSON for detection/segmentation training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = [
        {"id": i + 1, "name": name}
        for i, name in enumerate(DETECTION_CLASS_NAMES)
        if name != "background"
    ]
    name_to_cat = {c["name"]: c["id"] for c in categories}
    # Map micrococci etc.; background samples skipped for detection boxes
    images = []
    annotations = []
    ann_id = 1
    for img_id, rec in enumerate(records, start=1):
        if split is not None and rec.split != split:
            continue
        if rec.class_name == "background":
            # Still list the image for negative mining if desired
            images.append(
                {
                    "id": img_id,
                    "file_name": rec.image_path,
                    "width": rec.width or 0,
                    "height": rec.height or 0,
                }
            )
            continue
        cat = name_to_cat.get(rec.class_name)
        if cat is None:
            continue
        images.append(
            {
                "id": img_id,
                "file_name": rec.image_path,
                "width": rec.width or 0,
                "height": rec.height or 0,
            }
        )
        if rec.bbox is None:
            continue
        x0, y0, x1, y1 = rec.bbox
        w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
        annotations.append(
            {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat,
                "bbox": [x0, y0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [],
            }
        )
        ann_id += 1
    payload = {"images": images, "annotations": annotations, "categories": categories}
    path.write_text(json.dumps(payload), encoding="utf-8")
    logger.info("Wrote COCO json %s (%d images, %d anns)", path, len(images), len(annotations))


def iter_balanced_batches(
    records: Sequence[SampleRecord],
    batch_size: int,
    *,
    seed: int = 0,
) -> Iterator[list[SampleRecord]]:
    """Yield class-balanced mini-batches (round-robin over classes)."""
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[SampleRecord]] = {}
    for r in records:
        by_class.setdefault(r.class_name, []).append(r)
    for lst in by_class.values():
        rng.shuffle(lst)
    classes = sorted(by_class.keys())
    pointers = {c: 0 for c in classes}
    while True:
        batch: list[SampleRecord] = []
        while len(batch) < batch_size:
            progress = False
            for c in classes:
                lst = by_class[c]
                if not lst:
                    continue
                idx = pointers[c] % len(lst)
                batch.append(lst[idx])
                pointers[c] += 1
                progress = True
                if len(batch) >= batch_size:
                    break
            if not progress:
                return
        yield batch
