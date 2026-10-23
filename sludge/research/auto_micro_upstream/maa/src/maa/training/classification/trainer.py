"""Classification training loop (512², AdamW cosine, AMP)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from maa.constants import DEFAULT_EPOCHS, EXPERIMENT_SEEDS, INPUT_SIZE
from maa.data.collate import collate_classification
from maa.data.dataset_classification import MicroscopyClassificationDataset
from maa.data.transforms import TransformConfig, build_classification_transforms
from maa.models.classification.registry import build_classification_model
from maa.training.classification.losses import ClassificationLoss
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
class ClassificationTrainConfig:
    data_root: str = "."
    manifest: str | None = None
    split_manifest: str | None = None
    output_dir: str = "runs/cls"
    backbone: str = "efficientnetv2"
    attention: str = "maa"
    num_classes: int = 5
    batch_size: int = 16
    epochs: int = DEFAULT_EPOCHS
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 0
    num_workers: int = 4
    seed: int = EXPERIMENT_SEEDS[0]
    amp: bool = True
    patience: int = 10
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    input_size: int = INPUT_SIZE
    device: str | None = None
    pretrained: bool = True
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClassificationTrainConfig":
        raw = load_yaml_config(path)
        data = raw.get("data", {})
        model = raw.get("model", {})
        train = raw.get("train", {})
        return cls(
            data_root=str(data.get("root", ".")),
            manifest=data.get("manifest"),
            split_manifest=data.get("split_manifest"),
            output_dir=str(raw.get("output_dir", "runs/cls")),
            backbone=str(model.get("backbone", "efficientnetv2")),
            attention=str(model.get("attention", "maa")),
            num_classes=int(model.get("num_classes", 5)),
            batch_size=int(train.get("batch_size", 16)),
            epochs=int(train.get("epochs", DEFAULT_EPOCHS)),
            lr=float(train.get("lr", 1e-4)),
            weight_decay=float(train.get("weight_decay", 0.05)),
            warmup_epochs=int(train.get("warmup_epochs", 0)),
            num_workers=int(train.get("num_workers", 4)),
            seed=int(train.get("seed", EXPERIMENT_SEEDS[0])),
            amp=bool(train.get("amp", True)),
            patience=int(train.get("patience", 10)),
            use_class_weights=bool(train.get("use_class_weights", True)),
            label_smoothing=float(train.get("label_smoothing", 0.0)),
            input_size=int(data.get("input_size", INPUT_SIZE)),
            device=train.get("device"),
            pretrained=bool(model.get("pretrained", True)),
            extra_model_kwargs=dict(model.get("kwargs", {})),
        )


class ClassificationTrainer:
    def __init__(self, cfg: ClassificationTrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        tcfg = TransformConfig(input_size=cfg.input_size)

        self.train_ds = MicroscopyClassificationDataset(
            cfg.data_root,
            manifest=cfg.manifest,
            split="train",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_classification_transforms(True, tcfg, seed=cfg.seed),
        )
        self.val_ds = MicroscopyClassificationDataset(
            cfg.data_root,
            manifest=cfg.manifest,
            split="val",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_classification_transforms(False, tcfg),
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_classification,
        )
        self.val_loader = DataLoader(
            self.val_ds,
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
            pretrained=cfg.pretrained,
            disabled_prior=cfg.extra_model_kwargs.get("disabled_prior"),
        ).to(self.device)

        weights = self.train_ds.compute_class_weights() if cfg.use_class_weights else None
        self.criterion = ClassificationLoss(class_weights=weights, label_smoothing=cfg.label_smoothing)
        self.optimizer = build_optimizer(self.model, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = build_scheduler(
            self.optimizer,
            epochs=cfg.epochs,
            warmup_epochs=cfg.warmup_epochs,
            steps_per_epoch=len(self.train_loader),
        )
        self.amp = AMPContext(enabled=cfg.amp and self.device.type == "cuda")
        self.early_stop = EarlyStopping(patience=cfg.patience, mode="max")
        self.checkpoints = CheckpointManager(
            Path(cfg.output_dir) / f"seed_{cfg.seed}",
            monitor="val_acc",
            mode="max",
        )

        logger.info(
            "Classification trainer: %s + %s | params=%s",
            cfg.backbone,
            cfg.attention,
            f"{count_params(self.model):,}",
        )

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            with self.amp.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            loss_sum += float(loss) * labels.size(0)
        acc = correct / max(total, 1)
        return {"val_loss": loss_sum / max(total, 1), "val_acc": acc}

    def train(self) -> dict[str, Any]:
        history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for batch in pbar:
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.amp.scaler.scale(loss).backward()
                self.amp.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.amp.scaler.step(self.optimizer)
                self.amp.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                bs = labels.size(0)
                running_loss += float(loss) * bs
                n += bs
                pbar.set_postfix(loss=running_loss / max(n, 1))

            metrics = self.evaluate()
            metrics["train_loss"] = running_loss / max(n, 1)
            metrics["epoch"] = float(epoch)
            history.append(metrics)
            logger.info(
                "Epoch %d | train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                metrics["train_loss"],
                metrics["val_loss"],
                metrics["val_acc"],
            )

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "cfg": self.cfg.__dict__,
                "metrics": metrics,
            }
            self.checkpoints.save(state, epoch=epoch, metric=metrics["val_acc"], tag="last")
            improved = self.early_stop.step(metrics["val_acc"])
            if improved:
                self.checkpoints.save(state, epoch=epoch, metric=metrics["val_acc"], tag="best")
            if self.early_stop.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        latency = measure_latency(self.model, device=self.device)
        return {"history": history, "latency": latency, "best_val_acc": self.early_stop.best}
