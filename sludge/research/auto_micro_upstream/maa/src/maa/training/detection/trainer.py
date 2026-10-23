"""Detection training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from maa.constants import DEFAULT_EPOCHS, EXPERIMENT_SEEDS, INPUT_SIZE
from maa.data.collate import collate_detection
from maa.data.dataset_detection import COCODetectionDataset
from maa.data.transforms import TransformConfig, build_detection_transforms
from maa.models.detection.registry import build_detection_model
from maa.training.common import (
    AMPContext,
    CheckpointManager,
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    count_params,
    load_yaml_config,
    measure_latency,
    resolve_device,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionTrainConfig:
    data_root: str = "."
    annotation_file: str = "annotations/train.json"
    split_manifest: str | None = None
    output_dir: str = "runs/det"
    backbone: str = "yolov13"
    attention: str = "maa"
    num_classes: int = 5
    num_queries: int = 100
    batch_size: int = 8
    epochs: int = DEFAULT_EPOCHS
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 0
    num_workers: int = 4
    seed: int = EXPERIMENT_SEEDS[0]
    amp: bool = True
    patience: int = 10
    input_size: int = INPUT_SIZE
    device: str | None = None
    pretrained: bool = True
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DetectionTrainConfig":
        raw = load_yaml_config(path)
        data = raw.get("data", {})
        model = raw.get("model", {})
        train = raw.get("train", {})
        return cls(
            data_root=str(data.get("root", ".")),
            annotation_file=str(data.get("annotation_file", "annotations/train.json")),
            split_manifest=data.get("split_manifest"),
            output_dir=str(raw.get("output_dir", "runs/det")),
            backbone=str(model.get("backbone", "yolov13")),
            attention=str(model.get("attention", "maa")),
            num_classes=int(model.get("num_classes", 5)),
            num_queries=int(model.get("num_queries", 100)),
            batch_size=int(train.get("batch_size", 8)),
            epochs=int(train.get("epochs", DEFAULT_EPOCHS)),
            lr=float(train.get("lr", 1e-4)),
            weight_decay=float(train.get("weight_decay", 0.05)),
            warmup_epochs=int(train.get("warmup_epochs", 0)),
            num_workers=int(train.get("num_workers", 4)),
            seed=int(train.get("seed", EXPERIMENT_SEEDS[0])),
            amp=bool(train.get("amp", True)),
            patience=int(train.get("patience", 10)),
            input_size=int(data.get("input_size", INPUT_SIZE)),
            device=train.get("device"),
            pretrained=bool(model.get("pretrained", True)),
            extra_model_kwargs=dict(model.get("kwargs", {})),
        )


class DetectionTrainer:
    def __init__(self, cfg: DetectionTrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        tcfg = TransformConfig(input_size=cfg.input_size)

        self.train_ds = COCODetectionDataset(
            cfg.data_root,
            cfg.annotation_file,
            split="train",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_detection_transforms(True, tcfg, seed=cfg.seed),
        )
        self.val_ds = COCODetectionDataset(
            cfg.data_root,
            cfg.annotation_file,
            split="val",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_detection_transforms(False, tcfg),
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_detection,
        )
        self.val_loader = DataLoader(
            self.val_ds,
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
            pretrained=cfg.pretrained,
            disabled_prior=cfg.extra_model_kwargs.get("disabled_prior"),
        ).to(self.device)
        self.optimizer = build_optimizer(self.model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = build_scheduler(
            self.optimizer,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
            steps_per_epoch=len(self.train_loader),
        )
        self.amp = AMPContext(enabled=cfg.amp and self.device.type == "cuda")
        self.early_stop = EarlyStopping(patience=cfg.patience, mode="min")
        self.checkpoints = CheckpointManager(
            Path(cfg.output_dir) / f"seed_{cfg.seed}",
            monitor="val_loss",
            mode="min",
        )
        logger.info("Detection trainer params=%s", f"{count_params(self.model):,}")

    def _batch_targets(
        self,
        boxes_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Pack variable-length targets into the host model's expected batch dict."""
        batch_boxes: list[torch.Tensor] = []
        batch_labels: list[torch.Tensor] = []
        for boxes, labels in zip(boxes_list, labels_list):
            if boxes.numel() == 0:
                batch_boxes.append(torch.zeros(4, device=self.device))
                batch_labels.append(torch.zeros((), dtype=torch.long, device=self.device))
            else:
                batch_boxes.append(boxes[0].to(self.device))
                batch_labels.append(labels[0].to(self.device))
        return {
            "boxes": torch.stack(batch_boxes),
            "labels": torch.stack(batch_labels),
        }

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            targets = self._batch_targets(batch["boxes"], batch["labels"])
            with self.amp.autocast():
                self.model.train(True)
                outputs = self.model(images, targets)
                loss = outputs["loss_total"]
                self.model.eval()
            bs = images.size(0)
            total_loss += float(loss) * bs
            n += bs
        return {"val_loss": total_loss / max(n, 1)}

    def train(self) -> dict[str, Any]:
        history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for batch in pbar:
                images = batch["images"].to(self.device)
                targets = self._batch_targets(batch["boxes"], batch["labels"])
                self.optimizer.zero_grad(set_to_none=True)
                with self.amp.autocast():
                    self.model.train(True)
                    outputs = self.model(images, targets)
                    loss = outputs["loss_total"]
                self.amp.scaler.scale(loss).backward()
                self.amp.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.amp.scaler.step(self.optimizer)
                self.amp.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                bs = images.size(0)
                running_loss += float(loss) * bs
                n += bs
                pbar.set_postfix(loss=running_loss / max(n, 1))

            metrics = self.evaluate()
            metrics["train_loss"] = running_loss / max(n, 1)
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f",
                epoch,
                metrics["train_loss"],
                metrics["val_loss"],
            )

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg.__dict__,
                "metrics": metrics,
            }
            self.checkpoints.save(state, epoch=epoch, metric=metrics["val_loss"], tag="last")
            if self.early_stop.step(metrics["val_loss"]):
                self.checkpoints.save(state, epoch=epoch, metric=metrics["val_loss"], tag="best")
            if self.early_stop.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        latency = measure_latency(self.model, device=self.device)
        return {"history": history, "latency": latency, "best_val_loss": self.early_stop.best}
