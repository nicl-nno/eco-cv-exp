"""Base class for detection hosts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.constants import DETECTION_CLASS_NAMES, NUM_CLASSES
from maa.models.adapters import build_attention_adapter
from maa.attention.registry import is_maa_variant


@dataclass
class DetectionOutput:
    """Inference-time detection bundle."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {"boxes": self.boxes, "scores": self.scores, "labels": self.labels}


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class FPNNeck(nn.Module):
    """Feature-pyramid neck fusing multi-scale backbone outputs (P3–P5)."""

    def __init__(self, in_channels: tuple[int, ...], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [ConvBNAct(out_channels, out_channels, 3) for _ in in_channels]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        return [sm(x) for sm, x in zip(self.smooth, laterals)]


class DetectionHead(nn.Module):
    """Shared conv head predicting class logits and box deltas per anchor cell."""

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        hidden = max(in_channels, 128)
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, hidden, 3),
            ConvBNAct(hidden, hidden, 3),
        )
        self.cls_head = nn.Conv2d(hidden, num_anchors * num_classes, 3, padding=1)
        self.box_head = nn.Conv2d(hidden, num_anchors * 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        return self.cls_head(x), self.box_head(x)


class DetectionModelBase(nn.Module, ABC):
    """
    Detection host: backbone stem, FPN neck, attention at fusion, and class/box head.

    Training mode expects ``targets`` dict with ``boxes`` and ``labels``; returns
    a loss dictionary. Eval mode decodes predictions into ``boxes``/``scores``/``labels``.
    """

    backbone_name: str = "detector"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = len(DETECTION_CLASS_NAMES),
        fpn_channels: int = 256,
        disabled_prior: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.attention_name = attention.lower().strip()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self._uses_maa = is_maa_variant(self.attention_name) or self.attention_name.startswith("maa")

        if self.attention_name in ("none", "no", "identity", "-"):
            self.fusion_attention: nn.Module = nn.Identity()
        else:
            self.fusion_attention = build_attention_adapter(
                self.attention_name,
                fpn_channels,
                disabled_prior=disabled_prior,
            )

        self.cls_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.SmoothL1Loss(reduction="none")

    @abstractmethod
    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return multi-scale feature maps from the host stem."""

    @abstractmethod
    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return fused P-level map after neck + attention."""

    def decode_predictions(
        self,
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        *,
        score_thresh: float = 0.05,
        topk: int = 100,
    ) -> DetectionOutput:
        """Simple centre-based decoding for baseline inference."""
        b, _, h, w = cls_logits.shape
        cls_probs = cls_logits.softmax(dim=1)
        scores, labels = cls_probs.max(dim=1)
        scores = scores.view(b, -1)
        labels = labels.view(b, -1)
        boxes = box_deltas.permute(0, 2, 3, 1).reshape(b, h * w, 4).sigmoid()

        batch_boxes, batch_scores, batch_labels = [], [], []
        for i in range(b):
            mask = scores[i] > score_thresh
            s = scores[i][mask]
            if s.numel() == 0:
                batch_boxes.append(torch.zeros(0, 4, device=cls_logits.device))
                batch_scores.append(torch.zeros(0, device=cls_logits.device))
                batch_labels.append(torch.zeros(0, dtype=torch.long, device=cls_logits.device))
                continue
            k = min(topk, s.numel())
            top_idx = torch.topk(s, k=k).indices
            valid = mask.nonzero(as_tuple=False).squeeze(1)[top_idx]
            batch_boxes.append(boxes[i, valid])
            batch_scores.append(scores[i, valid])
            batch_labels.append(labels[i, valid])
        return DetectionOutput(
            boxes=torch.stack([b if b.numel() else torch.zeros(1, 4, device=cls_logits.device) for b in batch_boxes]),
            scores=torch.stack([s if s.numel() else torch.zeros(1, device=cls_logits.device) for s in batch_scores]),
            labels=torch.stack([l if l.numel() else torch.zeros(1, dtype=torch.long, device=cls_logits.device) for l in batch_labels]),
        )

    def compute_losses(
        self,
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Set-prediction style classification and box regression losses."""
        labels = targets["labels"].long()
        boxes = targets["boxes"].float()
        b = cls_logits.shape[0]
        cls_target = labels[:, 0].clamp(0, self.num_classes - 1)
        cls_loss = self.cls_loss(
            cls_logits.mean(dim=(2, 3)),
            cls_target,
        ).mean()
        pred_boxes = box_deltas.mean(dim=(2, 3))
        tgt_boxes = boxes[:, 0] if boxes.ndim == 3 else boxes
        if tgt_boxes.shape[-1] != 4:
            tgt_boxes = tgt_boxes[..., :4]
        box_loss = self.box_loss(pred_boxes, tgt_boxes).mean()
        return {"loss_cls": cls_loss, "loss_box": box_loss, "loss_total": cls_loss + box_loss}

    def apply_fusion_attention(self, fused: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if self._uses_maa:
            return self.fusion_attention(fused, images)
        return self.fusion_attention(fused)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, Any]:
        fused = self.forward_features(images)
        cls_logits, box_deltas = self.det_head(fused)
        if self.training and targets is not None:
            return self.compute_losses(cls_logits, box_deltas, targets)
        out = self.decode_predictions(cls_logits, box_deltas)
        return out.as_dict()


class MultiScaleStem(nn.Module):
    """Generic CNN stem emitting three pyramid levels."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stage1 = nn.Sequential(ConvBNAct(3, c1, 3, s=2), ConvBNAct(c1, c1, 3))
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, 3, s=2), ConvBNAct(c2, c2, 3))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, s=2), ConvBNAct(c3, c3, 3))
        self.out_channels = (c1, c2, c3)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return [p3, p4, p5]
