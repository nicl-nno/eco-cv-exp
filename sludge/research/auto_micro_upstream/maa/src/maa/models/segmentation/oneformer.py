"""OneFormer unified segmentation host (Jain et al., 2023)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.segmentation.base import ConvDecoder, ConvEncoder, SegmentationModelBase


class TaskTokenEncoder(nn.Module):
    """Task-conditioned token for semantic / instance / panoptic unification."""

    def __init__(self, dim: int = 256, num_tasks: int = 3) -> None:
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=dim * 4, batch_first=True),
            num_layers=2,
        )

    def forward(self, features: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        b, c, h, w = features.shape
        task = self.task_embed.weight[task_id].view(1, 1, -1).expand(b, h * w, -1)
        memory = features.flatten(2).transpose(1, 2)
        tokens = self.transformer(memory + task)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class OneFormerSegmenter(SegmentationModelBase):
    """
    OneFormer task-unified transformer segmenter.
    Default task token selects semantic segmentation for microscopy benchmark.
    """

    backbone_name = "oneformer"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        decoder_channels: int = 256,
        task_id: int = 0,
    ) -> None:
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            decoder_channels=decoder_channels,
            disabled_prior=disabled_prior,
        )
        self.task_id = task_id
        self.encoder = ConvEncoder(base=64)
        self.proj = nn.Conv2d(self.encoder.out_channels, decoder_channels, 1)
        self.task_encoder = TaskTokenEncoder(dim=decoder_channels)
        self.decoder = ConvDecoder(decoder_channels, decoder_channels, stages=3)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.proj(self.encoder(images))
        return self.task_encoder(feats, task_id=self.task_id)

    def decode(self, features: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return self.decoder(features, out_size)


def build_oneformer(
    *,
    attention: str = "none",
    num_classes: int = NUM_CLASSES,
    disabled_prior: Optional[str] = None,
) -> OneFormerSegmenter:
    return OneFormerSegmenter(
        attention=attention,
        num_classes=num_classes,
        disabled_prior=disabled_prior,
    )
