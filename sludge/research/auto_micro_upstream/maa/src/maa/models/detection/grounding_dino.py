"""Grounding DINO open-vocabulary detection host."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from maa.constants import DETECTION_CLASS_NAMES, GROUNDING_DINO_PROMPT
from maa.models.detection.base import (
    DetectionHead,
    DetectionModelBase,
    FPNNeck,
    MultiScaleStem,
)


class TextPromptEncoder(nn.Module):
    """
    Lightweight text encoder for microscopy open-vocab prompts.
    Uses ``GROUNDING_DINO_PROMPT`` token statistics when no external LM is loaded.
    """

    def __init__(self, vocab_size: int = 512, dim: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("_default_prompt_ids", self._tokenize(GROUNDING_DINO_PROMPT), persistent=False)

    @staticmethod
    def _tokenize(text: str, max_len: int = 32) -> torch.Tensor:
        tokens = [ord(c) % 512 for c in text.lower().strip()]
        tokens = tokens[:max_len]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def encode(self, prompts: Optional[list[str]] = None, device: torch.device | None = None) -> torch.Tensor:
        if prompts is None:
            ids = self._default_prompt_ids
            if device is not None:
                ids = ids.to(device)
            return self.proj(self.embedding(ids).mean(dim=0, keepdim=True))
        batch_ids = torch.stack([self._tokenize(p) for p in prompts])
        if device is not None:
            batch_ids = batch_ids.to(device)
        return self.proj(self.embedding(batch_ids).mean(dim=1))

    def forward(self, prompts: Optional[list[str]] = None) -> torch.Tensor:
        return self.encode(prompts)


class CrossModalFusion(nn.Module):
    """Bidirectional vision-language fusion (Grounding DINO style)."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.v2t = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.t2v = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)

    def forward(self, vision: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        b, c, h, w = vision.shape
        v_tokens = vision.flatten(2).transpose(1, 2)
        t_tokens = text.unsqueeze(1).expand(b, v_tokens.shape[1], -1)
        v2t, _ = self.v2t(self.norm_v(v_tokens), self.norm_t(t_tokens), t_tokens)
        v_tokens = v_tokens + v2t
        t2v, _ = self.t2v(self.norm_t(t_tokens), self.norm_v(v_tokens), v_tokens)
        v_tokens = v_tokens + t2v
        return v_tokens.transpose(1, 2).reshape(b, c, h, w)


class GroundingDINODetector(DetectionModelBase):
    """
    Grounding DINO detector for open-vocabulary microorganism localisation.
    Default text prompt from ``constants.GROUNDING_DINO_PROMPT``.
    """

    backbone_name = "grounding_dino"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = len(DETECTION_CLASS_NAMES),
        disabled_prior: Optional[str] = None,
        fpn_channels: int = 256,
        text_prompt: str = GROUNDING_DINO_PROMPT,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            fpn_channels=fpn_channels,
            disabled_prior=disabled_prior,
        )
        self.text_prompt = text_prompt
        self.backbone = MultiScaleStem(base_channels=64)
        self.neck = FPNNeck(self.backbone.out_channels, fpn_channels)
        self.text_encoder = TextPromptEncoder(dim=fpn_channels)
        self.cross_modal = CrossModalFusion(dim=fpn_channels)
        self.det_head = DetectionHead(fpn_channels, num_classes)

    def forward_backbone(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.backbone(images)

    def forward_features(
        self,
        images: torch.Tensor,
        text_prompts: Optional[list[str]] = None,
    ) -> torch.Tensor:
        feats = self.neck(self.forward_backbone(images))
        fused = feats[1]
        prompts = text_prompts or [self.text_prompt]
        text_emb = self.text_encoder.encode(prompts, device=images.device)
        fused = self.cross_modal(fused, text_emb)
        return self.apply_fusion_attention(fused, images)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[dict[str, torch.Tensor]] = None,
        text_prompts: Optional[list[str]] = None,
    ) -> dict:
        fused = self.forward_features(images, text_prompts=text_prompts)
        cls_logits, box_deltas = self.det_head(fused)
        if self.training and targets is not None:
            return self.compute_losses(cls_logits, box_deltas, targets)
        return self.decode_predictions(cls_logits, box_deltas).as_dict()


def build_grounding_dino(
    *,
    attention: str = "none",
    num_classes: int = len(DETECTION_CLASS_NAMES),
    disabled_prior: Optional[str] = None,
    text_prompt: str = GROUNDING_DINO_PROMPT,
) -> GroundingDINODetector:
    return GroundingDINODetector(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
        text_prompt=text_prompt,
    )
