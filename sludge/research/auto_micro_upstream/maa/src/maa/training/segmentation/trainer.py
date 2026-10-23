"""Segmentation training loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from maa.constants import DEFAULT_EPOCHS, EXPERIMENT_SEEDS, INPUT_SIZE, NUM_CLASSES
from maa.data.collate import collate_segmentation
from maa.data.dataset_segmentation import MicroscopySegmentationDataset
from maa.data.transforms import TransformConfig, build_segmentation_transforms
from maa.models.segmentation.registry import build_segmentation_model
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
from maa.training.segmentation.losses import SegmentationLoss, dice_coefficient

logger = logging.getLogger(__name__)


@dataclass
class SegmentationTrainConfig:
    data_root: str = "."
    split_manifest: str | None = None
    output_dir: str = "runs/seg"
    backbone: str = "mask2former"
    attention: str = "maa"
    num_classes: int = NUM_CLASSES
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
    ignore_index: int = 255
    device: str | None = None
    pretrained: bool = True
    extra_model_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SegmentationTrainConfig":
        raw = load_yaml_config(path)
        data = raw.get("data", {})
        model = raw.get("model", {})
        train = raw.get("train", {})
        return cls(
            data_root=str(data.get("root", ".")),
            split_manifest=data.get("split_manifest"),
            output_dir=str(raw.get("output_dir", "runs/seg")),
            backbone=str(model.get("backbone", "mask2former")),
            attention=str(model.get("attention", "maa")),
            num_classes=int(model.get("num_classes", NUM_CLASSES)),
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
            ignore_index=int(data.get("ignore_index", 255)),
            device=train.get("device"),
            pretrained=bool(model.get("pretrained", True)),
            extra_model_kwargs=dict(model.get("kwargs", {})),
        )


class SegmentationTrainer:
    def __init__(self, cfg: SegmentationTrainConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        tcfg = TransformConfig(input_size=cfg.input_size)

        self.train_ds = MicroscopySegmentationDataset(
            cfg.data_root,
            split="train",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_segmentation_transforms(True, tcfg, seed=cfg.seed),
            ignore_index=cfg.ignore_index,
        )
        self.val_ds = MicroscopySegmentationDataset(
            cfg.data_root,
            split="val",
            split_manifest=cfg.split_manifest,
            seed=cfg.seed,
            transform=build_segmentation_transforms(False, tcfg),
            ignore_index=cfg.ignore_index,
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate_segmentation,
        )
        self.val_loader = DataLoader(
            self.val_ds,
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
            pretrained=cfg.pretrained,
            disabled_prior=cfg.extra_model_kwargs.get("disabled_prior"),
        ).to(self.device)

        self.criterion = SegmentationLoss(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
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
            monitor="val_dice",
            mode="max",
        )
        logger.info("Segmentation trainer params=%s", f"{count_params(self.model):,}")

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        dice_sum = 0.0
        n = 0
        for batch in self.val_loader:
            images = batch["images"].to(self.device)
            masks = batch["masks"].to(self.device)
            with self.amp.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                dice = dice_coefficient(
                    logits,
                    masks,
                    num_classes=self.cfg.num_classes,
                    ignore_index=self.cfg.ignore_index,
                )
            bs = images.size(0)
            total_loss += float(loss) * bs
            dice_sum += float(dice) * bs
            n += bs
        return {"val_loss": total_loss / max(n, 1), "val_dice": dice_sum / max(n, 1)}

    def train(self) -> dict[str, Any]:
        history: list[dict[str, float]] = []
        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n = 0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.cfg.epochs}", leave=False)
            for batch in pbar:
                images = batch["images"].to(self.device)
                masks = batch["masks"].to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self.amp.autocast():
                    logits = self.model(images)
                    if isinstance(logits, dict):
                        logits = logits.get("pred_masks", logits.get("logits", logits["out"]))
                    loss = self.criterion(logits, masks)
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
                "Epoch %d | train_loss=%.4f val_loss=%.4f val_dice=%.4f",
                epoch,
                metrics["train_loss"],
                metrics["val_loss"],
                metrics["val_dice"],
            )

            state = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "cfg": self.cfg.__dict__,
                "metrics": metrics,
            }
            self.checkpoints.save(state, epoch=epoch, metric=metrics["val_dice"], tag="last")
            if self.early_stop.step(metrics["val_dice"]):
                self.checkpoints.save(state, epoch=epoch, metric=metrics["val_dice"], tag="best")
            if self.early_stop.should_stop:
                logger.info("Early stopping at epoch %d", epoch)
                break

        latency = measure_latency(self.model, device=self.device)
        return {"history": history, "latency": latency, "best_val_dice": self.early_stop.best}
