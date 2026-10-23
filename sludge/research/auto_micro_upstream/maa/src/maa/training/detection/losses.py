"""Detection losses: focal classification + L1/GIoU box regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU for xyxy boxes."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + eps)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + eps)
    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[..., 0] * wh_c[..., 1]
    return iou - (area_c - union) / (area_c + eps)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    return loss.mean()


class DetectionLoss(nn.Module):
    """
    DETR-style loss: focal on class logits and L1 + GIoU on matched boxes.

    Models should return a dict with keys:
    - ``pred_logits`` (B, Q, C)
    - ``pred_boxes`` (B, Q, 4) normalized cxcywh in [0, 1]
    """

    def __init__(
        self,
        num_classes: int,
        *,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        l1_weight: float = 5.0,
        giou_weight: float = 2.0,
        eos_coef: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.eos_coef = eos_coef

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

    def _match_queries(
        self,
        pred_boxes: torch.Tensor,
        tgt_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy IoU matching per image."""
        if tgt_boxes.numel() == 0:
            return pred_boxes.new_zeros((0, 4)), pred_boxes.new_zeros((0,), dtype=torch.long)
        pred_xyxy = self._cxcywh_to_xyxy(pred_boxes)
        tgt_xyxy = tgt_boxes
        ious = box_iou(pred_xyxy, tgt_xyxy)
        best_pred = ious.argmax(dim=0)
        matched_pred = pred_boxes[best_pred]
        return matched_pred, best_pred

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        image_size: int,
    ) -> torch.Tensor:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        b, q, c = pred_logits.shape
        device = pred_logits.device

        cls_losses: list[torch.Tensor] = []
        box_losses: list[torch.Tensor] = []

        for i in range(b):
            tgt_boxes = targets[i]["boxes"].to(device)
            tgt_labels = targets[i]["labels"].to(device)

            if tgt_boxes.numel() > 0:
                tgt_boxes_norm = tgt_boxes.clone()
                tgt_boxes_norm[:, [0, 2]] /= float(image_size)
                tgt_boxes_norm[:, [1, 3]] /= float(image_size)
                cx = (tgt_boxes_norm[:, 0] + tgt_boxes_norm[:, 2]) / 2.0
                cy = (tgt_boxes_norm[:, 1] + tgt_boxes_norm[:, 3]) / 2.0
                w = tgt_boxes_norm[:, 2] - tgt_boxes_norm[:, 0]
                h = tgt_boxes_norm[:, 3] - tgt_boxes_norm[:, 1]
                tgt_cxcywh = torch.stack([cx, cy, w, h], dim=-1)
            else:
                tgt_cxcywh = pred_boxes.new_zeros((0, 4))

            target_cls = torch.zeros((q, c), device=device)
            if tgt_labels.numel() > 0:
                matched_pred, pred_idx = self._match_queries(pred_boxes[i], tgt_boxes)
                for j, lbl in enumerate(tgt_labels):
                    target_cls[pred_idx[j], int(lbl)] = 1.0
                pred_matched = pred_boxes[i][pred_idx]
                tgt_matched = tgt_cxcywh
                l1 = F.l1_loss(pred_matched, tgt_matched, reduction="mean")
                giou = 1.0 - torch.diag(
                    generalized_box_iou(
                        self._cxcywh_to_xyxy(pred_matched),
                        tgt_boxes.to(device),
                    )
                ).mean()
                box_losses.append(self.l1_weight * l1 + self.giou_weight * giou)
            else:
                box_losses.append(pred_boxes.new_zeros(()))

            cls_losses.append(
                sigmoid_focal_loss(
                    pred_logits[i],
                    target_cls,
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                )
            )

        cls_loss = torch.stack(cls_losses).mean() if cls_losses else pred_logits.sum() * 0.0
        box_loss = torch.stack(box_losses).mean() if box_losses else pred_logits.sum() * 0.0
        return cls_loss + box_loss
