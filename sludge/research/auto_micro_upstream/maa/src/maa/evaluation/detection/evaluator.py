"""Detection test evaluator."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from maa.constants import INPUT_SIZE
from maa.data.collate import collate_detection
from maa.data.dataset_detection import COCODetectionDataset
from maa.data.transforms import TransformConfig, build_detection_transforms
from maa.evaluation.detection.metrics import compute_map50, compute_mean_iou
from maa.evaluation.latency import benchmark_inference
from maa.models.detection.registry import build_detection_model
from maa.training.common import CheckpointManager, load_yaml_config, resolve_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class DetectionEvalConfig:
    data_root: str = "."
    annotation_file: str = "annotations/test.json"
    split_manifest: str | None = None
    checkpoint: str = "checkpoint_best.pt"
    backbone: str = "yolov13"
    attention: str = "maa"
    num_classes: int = 5
    num_queries: int = 100
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    input_size: int = INPUT_SIZE
    device: str | None = None
    score_threshold: float = 0.05
    disabled_prior: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectionEvalConfig":
        raw = load_yaml_config(path)
        data = raw.get("data", {})
        model = raw.get("model", {})
        eval_cfg = raw.get("eval", {})
        return cls(
            data_root=str(data.get("root", ".")),
            annotation_file=str(data.get("annotation_file", "annotations/test.json")),
            split_manifest=data.get("split_manifest"),
            checkpoint=str(eval_cfg.get("checkpoint", "checkpoint_best.pt")),
            backbone=str(model.get("backbone", "yolov13")),
            attention=str(model.get("attention", "maa")),
            num_classes=int(model.get("num_classes", 5)),
            num_queries=int(model.get("num_queries", 100)),
            batch_size=int(eval_cfg.get("batch_size", 8)),
            num_workers=int(eval_cfg.get("num_workers", 4)),
            seed=int(eval_cfg.get("seed", 42)),
            input_size=int(data.get("input_size", INPUT_SIZE)),
            device=eval_cfg.get("device"),
            score_threshold=float(eval_cfg.get("score_threshold", 0.05)),
        )


class DetectionEvaluator:
    def __init__(self, cfg: DetectionEvalConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        tcfg = TransformConfig(input_size=cfg.input_size)

        self.test_ds = COCODetectionDataset(
            cfg.data_root,
            cfg.annotation_file,
            split="test",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_detection_transforms(False, tcfg),
        )
        self.loader = DataLoader(
            self.test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_detection,
        )

        self.model = build_detection_model(
            backbone=cfg.backbone,
            attention=cfg.attention,
            num_classes=cfg.num_classes,
            pretrained=False,
            disabled_prior=cfg.disabled_prior,
        ).to(self.device)

        if Path(cfg.checkpoint).exists():
            state = CheckpointManager.load(cfg.checkpoint, map_location=self.device)
            self.model.load_state_dict(state["model"])

    @torch.no_grad()
    def run(self) -> dict[str, Any]:
        self.model.eval()
        predictions: list[dict[str, np.ndarray]] = []
        ground_truth: list[dict[str, np.ndarray]] = []

        for i, batch in enumerate(self.loader):
            images = batch["images"].to(self.device)
            if i == 0:
                benchmark_inference(self.model, images[:1], warmup=10, repeats=30)
            outputs = self.model(images)
            for j in range(images.size(0)):
                boxes = outputs["boxes"][j].detach().cpu().numpy()
                scores = outputs["scores"][j].detach().cpu().numpy()
                labels = outputs["labels"][j].detach().cpu().numpy()
                keep = scores > self.cfg.score_threshold
                predictions.append(
                    {
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": labels[keep],
                    }
                )
            for boxes, labels in zip(batch["boxes"], batch["labels"]):
                ground_truth.append(
                    {
                        "boxes": boxes.cpu().numpy(),
                        "labels": labels.cpu().numpy(),
                    }
                )

        map_metrics = compute_map50(predictions, ground_truth, self.cfg.num_classes)
        miou = compute_mean_iou(predictions, ground_truth, score_threshold=0.5)
        return {"map50": map_metrics["map50"], "mean_iou": miou, "ap_per_class": map_metrics["ap_per_class"]}
