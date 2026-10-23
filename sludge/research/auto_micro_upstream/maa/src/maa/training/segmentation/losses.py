"""Segmentation losses: cross-entropy + Dice."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    valid = targets != ignore_index
    dice_per_class: list[torch.Tensor] = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred = probs[:, cls][valid]
        tgt = (targets[valid] == cls).float()
        if tgt.numel() == 0:
            continue
        inter = (pred * tgt).sum()
        denom = pred.sum() + tgt.sum()
        dice_per_class.append((2.0 * inter + eps) / (denom + eps))
    if not dice_per_class:
        return logits.new_tensor(0.0)
    return torch.stack(dice_per_class).mean()


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=weight, ignore_index=self.ignore_index)
        dice = 1.0 - dice_coefficient(
            logits,
            targets,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        return self.ce_weight * ce + self.dice_weight * dice
