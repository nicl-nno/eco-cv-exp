"""Segmentation test evaluator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from maa.constants import INPUT_SIZE, NUM_CLASSES
from maa.data.collate import collate_segmentation
from maa.data.dataset_segmentation import MicroscopySegmentationDataset
from maa.data.transforms import TransformConfig, build_segmentation_transforms
from maa.evaluation.latency import benchmark_inference
from maa.evaluation.segmentation.metrics import compute_segmentation_metrics
from maa.models.segmentation.registry import build_segmentation_model
from maa.training.common import CheckpointManager, load_yaml_config, resolve_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class SegmentationEvalConfig:
    data_root: str = "."
    split_manifest: str | None = None
    checkpoint: str = "checkpoint_best.pt"
    backbone: str = "mask2former"
    attention: str = "maa"
    num_classes: int = NUM_CLASSES
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    input_size: int = INPUT_SIZE
    ignore_index: int = 255
    device: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SegmentationEvalConfig":
        raw = load_yaml_config(path)
        data = raw.get("data", {})
        model = raw.get("model", {})
        eval_cfg = raw.get("eval", {})
        return cls(
            data_root=str(data.get("root", ".")),
            split_manifest=data.get("split_manifest"),
            checkpoint=str(eval_cfg.get("checkpoint", "checkpoint_best.pt")),
            backbone=str(model.get("backbone", "mask2former")),
            attention=str(model.get("attention", "maa")),
            num_classes=int(model.get("num_classes", NUM_CLASSES)),
            batch_size=int(eval_cfg.get("batch_size", 8)),
            num_workers=int(eval_cfg.get("num_workers", 4)),
            seed=int(eval_cfg.get("seed", 42)),
            input_size=int(data.get("input_size", INPUT_SIZE)),
            ignore_index=int(data.get("ignore_index", 255)),
            device=eval_cfg.get("device"),
        )


class SegmentationEvaluator:
    def __init__(self, cfg: SegmentationEvalConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        tcfg = TransformConfig(input_size=cfg.input_size)

        self.test_ds = MicroscopySegmentationDataset(
            cfg.data_root,
            split="test",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_segmentation_transforms(False, tcfg),
            ignore_index=cfg.ignore_index,
        )
        self.loader = DataLoader(
            self.test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_segmentation,
        )

        self.model = build_segmentation_model(
            backbone=cfg.backbone,
            attention=cfg.attention,
            num_classes=cfg.num_classes,
            pretrained=False,
        ).to(self.device)

        if Path(cfg.checkpoint).exists():
            state = CheckpointManager.load(cfg.checkpoint, map_location=self.device)
            self.model.load_state_dict(state["model"])

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        self.model.eval()
        preds: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        for i, batch in enumerate(self.loader):
            images = batch["images"].to(self.device)
            masks = batch["masks"].cpu().numpy()
            if i == 0:
                benchmark_inference(self.model, images[:1], warmup=10, repeats=30)
            logits = self.model(images)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(list(pred))
            targets.extend(list(masks))

        metrics = compute_segmentation_metrics(
            np.stack(preds),
            np.stack(targets),
            num_classes=self.cfg.num_classes,
            ignore_index=self.cfg.ignore_index,
        )
        return metrics
