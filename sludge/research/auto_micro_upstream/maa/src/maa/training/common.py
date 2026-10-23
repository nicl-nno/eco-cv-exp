"""Shared training utilities: seeds, optimizer, scheduler, AMP, checkpoints."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from maa.constants import DEFAULT_EPOCHS, EXPERIMENT_SEEDS, INPUT_SIZE, LR_SCHEDULE, OPTIMIZER_NAME

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Reproducibility helper for seeds (42, 123, 321)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    defaults_path = path.parent / "default.yaml"
    if defaults_path.exists() and path.name != "default.yaml":
        with defaults_path.open(encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh) or {}
        return _deep_merge(defaults, cfg)
    return cfg


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    betas: tuple[float, float] = (0.9, 0.999),
    name: str = OPTIMIZER_NAME,
) -> torch.optim.Optimizer:
    if name.lower() != "adamw":
        raise ValueError(f"Expected AdamW, got {name!r}")
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = DEFAULT_EPOCHS,
    warmup_epochs: int = 0,
    schedule: str = LR_SCHEDULE,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if schedule.lower() != "cosine":
        raise ValueError(f"Expected cosine schedule, got {schedule!r}")

    if steps_per_epoch is not None and steps_per_epoch > 0:
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        schedulers = []
        milestones = []
        if warmup_steps > 0:
            schedulers.append(LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps))
            milestones.append(warmup_steps)
        schedulers.append(CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps)))
        return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones or [0])

    schedulers = []
    milestones = []
    if warmup_epochs > 0:
        schedulers.append(LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs))
        milestones.append(warmup_epochs)
    schedulers.append(CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs)))
    if len(schedulers) == 1:
        return schedulers[0]
    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


@dataclass
class AMPContext:
    enabled: bool = True
    scaler: GradScaler = field(default_factory=GradScaler)

    def autocast(self):
        return autocast(enabled=self.enabled)


class EarlyStopping:
    """Stop when validation metric does not improve for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = False
        if self.best is None:
            improved = True
        elif self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class CheckpointManager:
    """Save best and last checkpoints with optional metric tracking."""

    def __init__(self, output_dir: str | Path, *, monitor: str = "val_loss", mode: str = "min") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric: float | None = None

    def _is_better(self, metric: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def save(
        self,
        state: dict[str, Any],
        *,
        epoch: int,
        metric: float,
        tag: str = "last",
    ) -> Path:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(state, path)
        meta = {
            "epoch": epoch,
            "metric": metric,
            "monitor": self.monitor,
            "path": str(path),
        }
        (self.output_dir / f"checkpoint_{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if self._is_better(metric):
            self.best_metric = metric
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            (self.output_dir / "checkpoint_best.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            logger.info("New best %s=%.6f saved to %s", self.monitor, metric, best_path)
        return path

    @staticmethod
    def load(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        return torch.load(path, map_location=map_location, weights_only=False)


def measure_latency(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, INPUT_SIZE, INPUT_SIZE),
    *,
    device: torch.device | None = None,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    """Optional inference latency measurement in milliseconds."""
    device = device or resolve_device()
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats

    return {
        "latency_ms": elapsed_ms,
        "params": float(count_params(model)),
        "input_size": float(input_size[-1]),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device, non_blocking=True) for t in v]
        else:
            out[k] = v
    return out


def run_epoch_loop(
    *,
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
    train: bool,
    forward_fn: Callable[[nn.Module, dict[str, Any]], torch.Tensor],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    amp: AMPContext | None = None,
    max_grad_norm: float | None = 1.0,
) -> float:
    """Generic epoch runner returning mean loss."""
    model.train(mode=train)
    losses: list[float] = []
    amp = amp or AMPContext(enabled=False)

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with amp.autocast():
            loss = forward_fn(model, batch)

        if train:
            amp.scaler.scale(loss).backward()
            if max_grad_norm is not None:
                amp.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()
            if scheduler is not None:
                scheduler.step()

        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else 0.0
