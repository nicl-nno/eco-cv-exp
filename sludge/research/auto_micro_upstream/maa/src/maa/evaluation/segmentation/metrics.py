"""Segmentation metrics: mean Dice and mean IoU."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _per_class_iou_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> tuple[list[float], list[float]]:
    ious: list[float] = []
    dices: list[float] = []
    valid = target != ignore_index
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_c = (pred == cls) & valid
        tgt_c = (target == cls) & valid
        inter = np.logical_and(pred_c, tgt_c).sum()
        union = np.logical_or(pred_c, tgt_c).sum()
        pred_sum = pred_c.sum()
        tgt_sum = tgt_c.sum()
        if union == 0 and pred_sum + tgt_sum == 0:
            continue
        ious.append(float(inter + eps) / float(union + eps))
        dices.append(float(2 * inter + eps) / float(pred_sum + tgt_sum + eps))
    return ious, dices


def compute_segmentation_metrics(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, Any]:
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    all_ious: list[float] = []
    all_dices: list[float] = []
    for pred, tgt in zip(predictions, targets):
        ious, dices = _per_class_iou_dice(pred, tgt, num_classes, ignore_index=ignore_index)
        all_ious.extend(ious)
        all_dices.extend(dices)

    return {
        "miou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mdice": float(np.mean(all_dices)) if all_dices else 0.0,
        "num_samples": int(len(predictions)),
    }
