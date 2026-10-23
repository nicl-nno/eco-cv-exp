"""Batch collation for classification, detection, and segmentation."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch


def collate_classification(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "images": images,
        "labels": labels,
        "paths": [b.get("path", "") for b in batch],
        "class_names": [b.get("class_name", "") for b in batch],
    }


def collate_detection(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "images": images,
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "image_ids": torch.tensor([b.get("image_id", i) for i, b in enumerate(batch)], dtype=torch.long),
        "paths": [b.get("path", "") for b in batch],
        "orig_sizes": torch.stack([b.get("orig_size", torch.zeros(2, dtype=torch.long)) for b in batch]),
    }


def collate_segmentation(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    return {
        "images": images,
        "masks": masks,
        "stems": [b.get("stem", "") for b in batch],
        "paths": [b.get("path", "") for b in batch],
    }


COLLATE_FNS: dict[str, Callable[[Sequence[dict[str, Any]]], dict[str, Any]]] = {
    "classification": collate_classification,
    "detection": collate_detection,
    "segmentation": collate_segmentation,
}


def build_collate_fn(task: str) -> Callable[[Sequence[dict[str, Any]]], dict[str, Any]]:
    key = task.lower().strip()
    if key not in COLLATE_FNS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(COLLATE_FNS)}")
    return COLLATE_FNS[key]
