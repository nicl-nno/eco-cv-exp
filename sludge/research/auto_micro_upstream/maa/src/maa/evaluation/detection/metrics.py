"""Detection metrics: mAP@0.5 and mean IoU."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def box_iou_xyxy(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    n, m = len(boxes1), len(boxes2)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32)
    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0) * np.maximum(boxes1[:, 3] - boxes1[:, 1], 0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0) * np.maximum(boxes2[:, 3] - boxes2[:, 1], 0)
    ious = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        xx1 = np.maximum(boxes1[i, 0], boxes2[:, 0])
        yy1 = np.maximum(boxes1[i, 1], boxes2[:, 1])
        xx2 = np.minimum(boxes1[i, 2], boxes2[:, 2])
        yy2 = np.minimum(boxes1[i, 3], boxes2[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = area1[i] + area2 - inter
        ious[i] = inter / (union + eps)
    return ious


def _match_predictions(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return TP/FP flags sorted by descending score for one image."""
    if len(pred_boxes) == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=bool)
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    pred_scores = pred_scores[order]

    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(len(pred_boxes), dtype=bool)
    fp = np.zeros(len(pred_boxes), dtype=bool)

    for i, (box, lbl) in enumerate(zip(pred_boxes, pred_labels)):
        candidates = np.where((gt_labels == lbl) & (~gt_matched))[0]
        if len(candidates) == 0:
            fp[i] = True
            continue
        ious = box_iou_xyxy(box[None, :], gt_boxes[candidates])[0]
        best = int(np.argmax(ious))
        best_iou = ious[best]
        gt_idx = candidates[best]
        if best_iou >= iou_threshold:
            tp[i] = True
            gt_matched[gt_idx] = True
        else:
            fp[i] = True
    return tp, fp


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    idx = np.where(recalls[1:] != recalls[:-1])[0]
    return float(np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1]))


def compute_map50(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    num_classes: int,
    score_threshold: float = 0.05,
) -> dict[str, float]:
    """
    COCO-style mAP@0.5 when pycocotools is unavailable.

    Each prediction dict: boxes (N,4), scores (N,), labels (N,)
    Each ground truth dict: boxes (M,4), labels (M,)
    """
    aps: list[float] = []
    for cls in range(num_classes):
        scores_all: list[float] = []
        tp_all: list[bool] = []
        fp_all: list[bool] = []
        n_gt = 0
        for pred, gt in zip(predictions, ground_truth):
            gt_mask = gt["labels"] == cls
            gt_boxes = gt["boxes"][gt_mask]
            n_gt += len(gt_boxes)
            pred_mask = (pred["labels"] == cls) & (pred["scores"] >= score_threshold)
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            pred_labels = pred["labels"][pred_mask]
            tp, fp = _match_predictions(
                pred_boxes,
                pred_scores,
                pred_labels,
                gt_boxes,
                np.full(len(gt_boxes), cls, dtype=int),
                iou_threshold=0.5,
            )
            scores_all.extend(pred_scores.tolist())
            tp_all.extend(tp.tolist())
            fp_all.extend(fp.tolist())

        if n_gt == 0:
            continue
        if len(scores_all) == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-np.array(scores_all))
        tp_cum = np.cumsum(np.array(tp_all)[order])
        fp_cum = np.cumsum(np.array(fp_all)[order])
        recalls = tp_cum / n_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        aps.append(average_precision(recalls, precisions))

    map50 = float(np.mean(aps)) if aps else 0.0
    return {"map50": map50, "ap_per_class": aps}


def compute_mean_iou(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    score_threshold: float = 0.5,
    match_iou: float = 0.5,
) -> float:
    ious: list[float] = []
    for pred, gt in zip(predictions, ground_truth):
        if len(gt["boxes"]) == 0:
            continue
        keep = pred["scores"] >= score_threshold
        pboxes = pred["boxes"][keep]
        plabels = pred["labels"][keep]
        for gbox, glbl in zip(gt["boxes"], gt["labels"]):
            same = plabels == glbl
            if not np.any(same):
                ious.append(0.0)
                continue
            iou_row = box_iou_xyxy(gbox[None, :], pboxes[same])[0]
            ious.append(float(np.max(iou_row)) if len(iou_row) else 0.0)
    return float(np.mean(ious)) if ious else 0.0


def decode_detection_outputs(
    outputs: dict[str, torch.Tensor],
    image_size: int,
    score_threshold: float = 0.05,
) -> list[dict[str, np.ndarray]]:
    """Convert model dict outputs to numpy predictions per batch item."""
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    batch_preds: list[dict[str, np.ndarray]] = []
    for i in range(logits.size(0)):
        scores, labels = logits[i].max(dim=-1)
        mask = scores >= score_threshold
        cx, cy, w, h = boxes[i][mask].unbind(-1)
        x1 = (cx - 0.5 * w) * image_size
        y1 = (cy - 0.5 * h) * image_size
        x2 = (cx + 0.5 * w) * image_size
        y2 = (cy + 0.5 * h) * image_size
        xyxy = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
        batch_preds.append(
            {
                "boxes": xyxy,
                "scores": scores[mask].cpu().numpy(),
                "labels": labels[mask].cpu().numpy(),
            }
        )
    return batch_preds
