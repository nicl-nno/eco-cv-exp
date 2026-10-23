"""Base class for classification hosts with optional attention injection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from maa.constants import NUM_CLASSES
from maa.models.adapters import build_attention_adapter
from maa.attention.registry import is_maa_variant


def _try_timm_backbone(
    timm_name: str,
    *,
    pretrained: bool = False,
    img_size: int = 512,
) -> tuple[Optional[nn.Module], Optional[int]]:
    """Optional external backbone hook; returns ``(None, None)`` by default."""
    del timm_name, pretrained, img_size
    return None, None


class ClassificationModelBase(nn.Module, ABC):
    """
    Wraps a CNN/ViT backbone, optional SE/SA/CBAM/MAA on the final feature map,
    and a linear classifier head.

    Subclasses implement ``extract_features`` returning ``(B, C, H, W)`` tensors
    suitable for ``build_attention_adapter``.
    """

    backbone_name: str = "base"

    def __init__(
        self,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        feature_channels: int,
        disabled_prior: Optional[str] = None,
        maa_dim: int = 256,
        maa_grid: int = 16,
    ) -> None:
        super().__init__()
        self.attention_name = attention.lower().strip()
        self.num_classes = num_classes
        self.feature_channels = feature_channels
        self._uses_maa = is_maa_variant(self.attention_name) or self.attention_name.startswith("maa")

        if self.attention_name in ("none", "no", "identity", "-"):
            self.attention: nn.Module = nn.Identity()
        else:
            self.attention = build_attention_adapter(
                self.attention_name,
                feature_channels,
                maa_dim=maa_dim,
                maa_grid=maa_grid,
                disabled_prior=disabled_prior,
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(feature_channels, num_classes)

    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return final spatial feature map ``(B, C, H, W)``."""

    def apply_attention(self, feats: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        if self._uses_maa:
            return self.attention(feats, images)
        return self.attention(feats)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images:
            ``(B, 3, H, W)`` RGB tensors normalised for the host backbone.

        Returns
        -------
        logits:
            ``(B, num_classes)`` unnormalised scores aligned with ``CLASS_NAMES``.
        """
        feats = self.extract_features(images)
        if feats.ndim == 3:
            # ViT-style (B, N, C) -> (B, C, H, W)
            b, n, c = feats.shape
            side = int(n**0.5)
            if side * side != n:
                raise ValueError(f"Token grid {n} is not square for {self.backbone_name}")
            feats = feats.transpose(1, 2).reshape(b, c, side, side)
        feats = self.apply_attention(feats, images)
        pooled = self.pool(feats).flatten(1)
        logits = self.head(self.dropout(pooled))
        return logits

    def extra_repr(self) -> str:
        return (
            f"backbone={self.backbone_name}, attention={self.attention_name!r}, "
            f"num_classes={self.num_classes}, channels={self.feature_channels}"
        )


class TimmClassificationHost(ClassificationModelBase):
    """Classification host wrapping an external feature backbone module."""

    def __init__(
        self,
        timm_name: str,
        feature_channels: int,
        *,
        attention: str = "none",
        num_classes: int = NUM_CLASSES,
        disabled_prior: Optional[str] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        if backbone is None:
            raise ValueError("backbone module required")
        resolved_channels = _resolve_feature_channels(backbone, feature_channels)
        super().__init__(
            attention=attention,
            num_classes=num_classes,
            feature_channels=resolved_channels,
            disabled_prior=disabled_prior,
        )
        self.backbone = backbone
        self.backbone_name = timm_name

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(images)
        else:
            feats = self.backbone(images)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            return feats
        if feats.ndim == 3:
            return feats
        raise ValueError(f"Unexpected feature shape {tuple(feats.shape)}")


def _resolve_feature_channels(model: nn.Module, default: int) -> int:
    """Resolve backbone output width with a short forward pass when metadata is ambiguous."""
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 128, 128)
            out = model.forward_features(dummy) if hasattr(model, "forward_features") else model(dummy)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            if out.ndim == 4:
                return int(out.shape[1])
            if out.ndim == 3:
                return int(out.shape[-1])
    except Exception:
        pass
    return default


class ConvStem(nn.Module):
    """CNN stem with successive stride-2 convolutions."""

    def __init__(self, out_channels: int = 256, stages: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 3
        for i in range(stages):
            out_ch = min(out_channels, 64 * (2**i))
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                ]
            )
            in_ch = out_ch
        self.stem = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class PatchTransformerEncoder(nn.Module):
    """Patch-embedding transformer encoder for ViT-style hosts."""

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        patch_size: int = 16,
        img_size: int = 512,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.embed_dim = embed_dim
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = self.grid * self.grid
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos_embed[:, : h * w]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)
