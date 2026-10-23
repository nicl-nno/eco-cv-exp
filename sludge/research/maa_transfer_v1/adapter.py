"""Isolated, fixed-frame Faster R-CNN/MAA research bridge; not a release model.

Uses the unmodified MIT-licensed auto_micro TokenAttentionAdapter at FPN P4.
The zero-initialized residual gate, inverse image normalization, and fixed-frame
guards are local adaptations, not claimed as the upstream benchmark recipe.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent / "auto_micro_upstream"
COMMIT = "6646311c98f409b1a2185fba26c5e223af359153"


def verify_upstream(root: Path = UPSTREAM) -> dict:
    """Verify the exact local upstream source snapshot before importing it."""
    manifest = json.loads((HERE / "upstream_manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("Unexpected upstream registration")
    actual = {p.relative_to(root).as_posix() for p in (root / "maa/src/maa").rglob("*.py")}
    if actual != set(manifest["files"]) - {"LICENSE"}:
        raise ValueError("Upstream Python source inventory changed")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Upstream checksum mismatch: {name}")
    return manifest


def upstream_adapter():
    verify_upstream()
    source = (UPSTREAM / "maa/src").resolve()
    if "maa" in sys.modules:
        if Path(sys.modules["maa"].__file__).resolve() != source / "maa/__init__.py":
            raise RuntimeError("A different maa package is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return importlib.import_module("maa.models.adapters").TokenAttentionAdapter


class MAAP4Backbone(nn.Module):
    """Refine P4 only; keep every other FPN output byte-for-byte unchanged."""

    def __init__(self, base: nn.Module, mean, std, *, priors_enabled: bool = True):
        super().__init__()
        if base.out_channels != 256:
            raise ValueError("v1 requires a 256-channel FPN")
        self.base = base
        self.out_channels = base.out_channels
        self.attention = upstream_adapter()(256, dim=256, grid=16, depth=1)
        for block in self.attention.blocks:
            for name in block.enable:
                block.set_prior_enabled(name, priors_enabled)
        self.priors_enabled = priors_enabled
        self.gate = nn.Parameter(torch.zeros(()))
        self.register_buffer("image_mean", torch.tensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(std).reshape(1, 3, 1, 1))

    def forward(self, normalized_images):
        features = self.base(normalized_images)
        if "2" not in features:
            raise ValueError("Missing FPN P4 (key '2')")
        # torchvision supplies normalized, resized images. The fixed-frame
        # outer wrapper guarantees no padding and no unknown valid-image mask.
        rgb = (normalized_images * self.image_std + self.image_mean).clamp(0, 1)
        original = features["2"]
        refined = self.attention(original, rgb)
        features["2"] = original + self.gate * (refined - original)
        return features


class FixedFrameMAADetector(nn.Module):
    """Explicitly unsupported for variable-sized/padded batches in v1.

    All BatchNorm running statistics are frozen, including new prior networks.
    No dataset, split, threshold selection, training loop or implicit downloads.
    """

    def __init__(
        self,
        detector: nn.Module,
        *,
        image_size: tuple[int, int] = (960, 1280),
        priors_enabled: bool = True,
    ):
        super().__init__()
        if len(image_size) != 2 or any(n < 32 or n % 32 for n in image_size):
            raise ValueError("Fixed image dimensions must be positive multiples of 32")
        if getattr(detector.transform, "fixed_size", None) is not None:
            raise ValueError("Unexpected fixed_size transform")
        if tuple(detector.transform.min_size) != (min(image_size),):
            raise ValueError("Transform min_size must preserve the fixed frame")
        if detector.transform.max_size != max(image_size):
            raise ValueError("Transform max_size must preserve the fixed frame")
        if detector.transform.size_divisible != 32:
            raise ValueError("Unexpected image padding stride")
        if isinstance(detector.backbone, MAAP4Backbone):
            raise ValueError("MAA is already installed")
        self.image_size = tuple(image_size)
        self.priors_enabled = priors_enabled
        detector.backbone = MAAP4Backbone(
            detector.backbone,
            detector.transform.image_mean,
            detector.transform.image_std,
            priors_enabled=priors_enabled,
        )
        self.detector = detector
        self.train(detector.training)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images, targets=None):
        if len(images) != 1 or tuple(images[0].shape) != (3, *self.image_size):
            raise ValueError("v1 supports one unpadded RGB frame of the registered size")
        if not torch.is_floating_point(images[0]):
            raise ValueError("RGB input must be floating point in [0,1]")
        if not torch.isfinite(images[0]).all() or images[0].min() < 0 or images[0].max() > 1:
            raise ValueError("RGB input must be finite and in [0,1]")
        return self.detector(images, targets)

    def export_metadata(self) -> dict:
        return {
            "format": "sludge_micro/maa-transfer-v1",
            "upstream_commit": COMMIT,
            "image_size": list(self.image_size),
            "fpn_level": "2",
            "token_grid": [16, 16],
            "prior_image_size": [512, 512],
            "priors_enabled": self.priors_enabled,
            "gate_initialization": 0.0,
            "batchnorm": "frozen",
            "production_ready": False,
        }
