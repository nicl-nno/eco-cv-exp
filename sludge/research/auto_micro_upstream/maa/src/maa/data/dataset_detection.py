"""Detection dataset backed by COCO-style JSON annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from maa.constants import DETECTION_CLASS_NAMES
from maa.data.splits import load_split_manifest, split_by_ids


@dataclass(frozen=True)
class DetectionTarget:
    boxes: torch.Tensor  # (N, 4) xyxy absolute pixels
    labels: torch.Tensor  # (N,) 1-based COCO category ids or 0-based internal ids


@dataclass(frozen=True)
class DetectionRecord:
    image_id: int
    path: Path
    width: int
    height: int
    target: DetectionTarget


def _build_category_maps(categories: list[dict[str, Any]]) -> tuple[dict[int, int], dict[int, str]]:
    """Map COCO category id -> contiguous label index."""
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(sorted_cats)}
    id_to_name = {c["id"]: c.get("name", str(c["id"])) for c in sorted_cats}
    return id_to_idx, id_to_name


def _coco_bbox_to_xyxy(bbox: list[float]) -> torch.Tensor:
    x, y, w, h = bbox
    return torch.tensor([x, y, x + w, y + h], dtype=torch.float32)


class COCODetectionDataset(Dataset):
    """
    COCO-like detection dataset.

    Expected JSON fields: ``images``, ``annotations``, ``categories``.
    Bounding boxes use COCO ``[x, y, width, height]`` format in annotation files
    and are converted to absolute ``xyxy`` for training transforms.
    """

    def __init__(
        self,
        root: str | Path,
        annotation_file: str | Path,
        *,
        split: str | None = None,
        split_manifest: str | Path | None = None,
        seed: int = 42,
        transform: Callable[[Any, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        | None = None,
        category_names: tuple[str, ...] = DETECTION_CLASS_NAMES,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.seed = seed
        self.category_names = category_names

        ann_path = Path(annotation_file)
        with ann_path.open(encoding="utf-8") as fh:
            coco = json.load(fh)

        self.id_to_idx, self.id_to_name = _build_category_maps(coco.get("categories", []))
        self.idx_to_name = {i: self.id_to_name[cid] for cid, i in self.id_to_idx.items()}

        anns_by_image: dict[int, list[dict[str, Any]]] = {}
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0):
                continue
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        records: list[DetectionRecord] = []
        for img in coco.get("images", []):
            image_id = int(img["id"])
            file_name = img["file_name"]
            path = Path(file_name)
            if not path.is_absolute():
                path = (self.root / file_name).resolve()
            width = int(img.get("width", 0))
            height = int(img.get("height", 0))

            boxes: list[torch.Tensor] = []
            labels: list[int] = []
            for ann in anns_by_image.get(image_id, []):
                cat_id = int(ann["category_id"])
                if cat_id not in self.id_to_idx:
                    continue
                boxes.append(_coco_bbox_to_xyxy(ann["bbox"]))
                labels.append(self.id_to_idx[cat_id])

            target = DetectionTarget(
                boxes=torch.stack(boxes) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                labels=torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            )
            records.append(
                DetectionRecord(
                    image_id=image_id,
                    path=path,
                    width=width,
                    height=height,
                    target=target,
                )
            )

        if not records:
            raise FileNotFoundError(f"No images found in {annotation_file}")

        if split_manifest is not None:
            split_map = load_split_manifest(split_manifest)
            allowed = {int(x) if str(x).isdigit() else x for x in split_map[split or "train"]}
            records = [r for r in records if r.image_id in allowed or str(r.path) in allowed]
        elif split is not None:
            ids = [r.image_id for r in records]
            buckets = split_by_ids(ids, seed=seed)
            allowed = set(buckets[split])
            records = [r for r in records if r.image_id in allowed]

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        with Image.open(rec.path) as img:
            image = img.convert("RGB")

        boxes = rec.target.boxes.clone()
        labels = rec.target.labels.clone()

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        return {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "image_id": rec.image_id,
            "path": str(rec.path),
            "orig_size": torch.tensor([rec.height, rec.width], dtype=torch.int64),
        }

    @property
    def num_classes(self) -> int:
        return len(self.id_to_idx)
