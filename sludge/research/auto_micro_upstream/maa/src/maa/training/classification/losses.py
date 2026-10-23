"""Classification losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    Cross-entropy with optional inverse-frequency class weights for long-tail
    microorganism taxonomy.
    """

    def __init__(
        self,
        *,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )
