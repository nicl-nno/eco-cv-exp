"""Classification test evaluator with latency warmup."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from maa.constants import CLASS_NAMES, INPUT_SIZE
from maa.data.collate import collate_classification
from maa.data.dataset_classification import MicroscopyClassificationDataset
from maa.data.transforms import TransformConfig, build_classification_transforms
from maa.evaluation.classification.metrics import compute_classification_metrics
from maa.evaluation.latency import benchmark_inference
from maa.models.classification.registry import build_classification_model
from maa.training.common import CheckpointManager, load_yaml_config, resolve_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class ClassificationEvalConfig:
    data_root: str = "."
    manifest: str | None = None
    split_manifest: str | None = None
    checkpoint: str = "checkpoint_best.pt"
    backbone: str = "efficientnetv2"
    attention: str = "maa"
    num_classes: int = 5
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    input_size: int = INPUT_SIZE
    device: str | None = None
    latency_warmup: int = 10
    latency_repeats: int = 50

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassificationEvalConfig":
        raw = load_yaml_config(path)
        data = raw.get("data", {})
        model = raw.get("model", {})
        eval_cfg = raw.get("eval", {})
        return cls(
            data_root=str(data.get("root", ".")),
            manifest=data.get("manifest"),
            split_manifest=data.get("split_manifest"),
            checkpoint=str(eval_cfg.get("checkpoint", "checkpoint_best.pt")),
            backbone=str(model.get("backbone", "efficientnetv2")),
            attention=str(model.get("attention", "maa")),
            num_classes=int(model.get("num_classes", 5)),
            batch_size=int(eval_cfg.get("batch_size", 16)),
            num_workers=int(eval_cfg.get("num_workers", 4)),
            seed=int(eval_cfg.get("seed", 42)),
            input_size=int(data.get("input_size", INPUT_SIZE)),
            device=eval_cfg.get("device"),
            latency_warmup=int(eval_cfg.get("latency_warmup", 10)),
            latency_repeats=int(eval_cfg.get("latency_repeats", 50)),
        )


class ClassificationEvaluator:
    def __init__(self, cfg: ClassificationEvalConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        tcfg = TransformConfig(input_size=cfg.input_size)

        self.test_ds = MicroscopyClassificationDataset(
            cfg.data_root,
            manifest=cfg.manifest,
            split="test",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_classification_transforms(False, tcfg),
        )
        self.loader = DataLoader(
            self.test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_classification,
        )

        self.model = build_classification_model(
            backbone=cfg.backbone,
            attention=cfg.attention,
            num_classes=cfg.num_classes,
            pretrained=False,
        ).to(self.device)

        if Path(cfg.checkpoint).exists():
            state = CheckpointManager.load(cfg.checkpoint, map_location=self.device)
            self.model.load_state_dict(state["model"])
            logger.info("Loaded checkpoint %s", cfg.checkpoint)

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        self.model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        latencies: list[float] = []

        for i, batch in enumerate(self.loader):
            images = batch["images"].to(self.device)
            labels = batch["labels"].cpu().numpy().tolist()

            if i == 0:
                benchmark_inference(
                    self.model,
                    images[:1],
                    warmup=self.cfg.latency_warmup,
                    repeats=self.cfg.latency_repeats,
                )

            start = time.perf_counter()
            logits = self.model(images)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0 / images.size(0))

            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            y_true.extend(labels)
            y_pred.extend(preds)

        metrics = compute_classification_metrics(
            np.array(y_true),
            np.array(y_pred),
            labels=list(range(self.cfg.num_classes)),
            target_names=list(CLASS_NAMES[: self.cfg.num_classes]),
        )
        metrics["latency_ms_per_image"] = float(np.mean(latencies)) if latencies else 0.0
        return metrics
