"""Unit tests for MAA priors, layers, zenodo helpers, and experiment grid."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from maa.attention.baselines.cbam import CBAMBlock
from maa.attention.baselines.sa import SABlock
from maa.attention.baselines.se import SEBlock
from maa.attention.maa.block import MAAConfig, MicroscopyAwareAttention
from maa.attention.maa.boundary import BoundaryPrior
from maa.attention.maa.compactness import CompactnessPrior
from maa.attention.maa.density import DensityPrior
from maa.attention.maa.focus import FocusPrior
from maa.attention.maa.morphology import MorphologyPrior
from maa.attention.maa.organelle import OrganellePrior
from maa.constants import ATTENTION_VARIANTS, CLASSIFICATION_BACKBONES
from maa.data.splits import assign_fixed_splits, stratified_split_indices
from maa.data.zenodo import SampleRecord, mask_to_bbox, write_manifest
from maa.models.classification.registry import build_classification_model
from maa.models.layers import FPN, MultiScaleEncoder, TransformerDecoder
from maa.training.experiments import build_experiment_grid, parse_config_stem


@pytest.fixture
def images() -> torch.Tensor:
    return torch.rand(2, 3, 64, 64)


def test_prior_shapes(images: torch.Tensor) -> None:
    n, th, tw = 64, 8, 8
    for prior in (
        BoundaryPrior(learnable_edge=False),
        FocusPrior(),
        OrganellePrior(),
        CompactnessPrior(),
        MorphologyPrior(),
        DensityPrior(use_mass=False),
    ):
        bias = prior(images, n, th, tw)
        assert bias.shape == (2, n, n)
        assert torch.isfinite(bias).all()


def test_maa_forward(images: torch.Tensor) -> None:
    block = MicroscopyAwareAttention(MAAConfig(dim=64, num_heads=4))
    tokens = torch.randn(2, 64, 64)
    out = block(tokens, images, tokens_h=8, tokens_w=8)
    assert out.shape == tokens.shape
    out2, extras = block(tokens, images, tokens_h=8, tokens_w=8, return_attn=True)
    assert out2.shape == tokens.shape
    assert "attn" in extras and "bias_parts" in extras
    assert set(extras["bias_parts"]) <= {
        "boundary",
        "focus",
        "organelle",
        "compactness",
        "morphology",
        "density",
    }


def test_maa_ablation_disables_prior(images: torch.Tensor) -> None:
    block = MicroscopyAwareAttention(MAAConfig(dim=32, num_heads=4, enable_morphology=False))
    tokens = torch.randn(2, 16, 32)
    _, extras = block(tokens, images, tokens_h=4, tokens_w=4, return_attn=True)
    assert "morphology" not in extras["bias_parts"]


def test_baseline_feature_attention(images: torch.Tensor) -> None:
    x = torch.randn(2, 32, 16, 16)
    assert SEBlock(32)(x).shape == x.shape
    assert SABlock()(x).shape == x.shape
    assert CBAMBlock(32)(x).shape == x.shape


def test_morphology_exact_patches(images: torch.Tensor) -> None:
    prior = MorphologyPrior(patch=5)
    scores = prior.token_scores(images, 16, 4, 4, exact_patches=True)
    assert scores.shape == (2, 16)
    bias = prior(images, 16, 4, 4)
    assert bias.shape == (2, 16, 16)


def test_mask_to_bbox() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:20, 10:18] = 255
    box = mask_to_bbox(mask)
    assert box is not None
    assert box[0] == 10 and box[1] == 8


def test_assign_fixed_splits() -> None:
    records = [
        SampleRecord(image_path=f"a/{i}.png", mask_path=None, class_name="bacilli", class_id=4)
        for i in range(20)
    ] + [
        SampleRecord(image_path=f"b/{i}.png", mask_path=None, class_name="micrococci", class_id=1)
        for i in range(20)
    ]
    out = assign_fixed_splits(records, seed=42)
    splits = {r.split for r in out}
    assert splits == {"train", "val", "test"}
    assert sum(1 for r in out if r.split == "train") >= 20


def test_manifest_roundtrip(tmp_path: Path) -> None:
    records = [
        SampleRecord("img.png", "mask.png", "bacilli", 4, split="train", bbox=[1, 2, 3, 4], width=10, height=10)
    ]
    path = tmp_path / "m.json"
    write_manifest(records, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["num_samples"] == 1


def test_layers_encoder_fpn_decoder() -> None:
    enc = MultiScaleEncoder(widths=(32, 64, 128), depth=1)
    x = torch.randn(1, 3, 128, 128)
    feats = enc(x)
    assert len(feats) == 3
    fpn = FPN([32, 64, 128], 64)
    outs = fpn(feats)
    assert all(o.shape[1] == 64 for o in outs)
    mem = outs[-1].flatten(2).transpose(1, 2)
    dec = TransformerDecoder(dim=64, depth=1, num_heads=4, num_queries=10, num_classes=4)
    cls, boxes = dec(mem)
    assert cls.shape == (1, 10, 5)
    assert boxes.shape == (1, 10, 4)


def test_classification_hosts_forward() -> None:
    x = torch.randn(1, 3, 128, 128)
    for bb in ("efficientnetv2", "convnext_v2", "swinv2"):
        for attn in ("none", "se", "maa"):
            model = build_classification_model(bb, attn, num_classes=5)
            model.eval()
            with torch.inference_mode():
                y = model(x)
            assert y.shape == (1, 5), f"{bb}/{attn}"


def test_parse_config_stem() -> None:
    assert parse_config_stem("efficientnetv2_maa") == ("efficientnetv2", "maa")
    assert parse_config_stem("rt_detr_v4_cbam") == ("rt_detr_v4", "cbam")
    assert parse_config_stem("grounding_dino_none") == ("grounding_dino", "none")


def test_experiment_grid_from_repo_configs() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    if not (root / "classification").exists():
        pytest.skip("configs not present")
    specs = build_experiment_grid(root, "classification", seeds=(42,))
    assert len(specs) == len(CLASSIFICATION_BACKBONES) * len(ATTENTION_VARIANTS)
    names = {(s.backbone, s.attention) for s in specs}
    assert ("dinov3", "maa") in names
