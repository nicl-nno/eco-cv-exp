"""Experiment orchestration: run all baseline configs for a task."""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from maa.constants import (
    ATTENTION_VARIANTS,
    CLASSIFICATION_BACKBONES,
    DETECTION_BACKBONES,
    EXPERIMENT_SEEDS,
    SEGMENTATION_BACKBONES,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSpec:
    task: str
    backbone: str
    attention: str
    config_path: str
    seed: int = 42
    output_dir: str = ""

    @property
    def run_name(self) -> str:
        return f"{self.task}/{self.backbone}_{self.attention}/seed_{self.seed}"


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    metrics: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0
    status: str = "ok"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.spec.task,
            "backbone": self.spec.backbone,
            "attention": self.spec.attention,
            "seed": self.spec.seed,
            "config_path": self.spec.config_path,
            "output_dir": self.spec.output_dir,
            "elapsed_sec": self.elapsed_sec,
            "status": self.status,
            "error": self.error,
            "metrics": self.metrics,
        }


def discover_baseline_configs(configs_root: str | Path, task: str) -> list[Path]:
    root = Path(configs_root) / task
    if not root.is_dir():
        raise FileNotFoundError(f"Missing config directory: {root}")
    return sorted(root.glob("*_*.yaml"))


def parse_config_stem(stem: str) -> tuple[str, str]:
    """Parse `{backbone}_{attention}` allowing multi-part backbone names."""
    for attn in sorted(ATTENTION_VARIANTS, key=len, reverse=True):
        suffix = f"_{attn}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], attn
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"Cannot parse backbone/attention from {stem!r}")


def build_experiment_grid(
    configs_root: str | Path,
    task: str,
    *,
    backbones: Sequence[str] | None = None,
    attentions: Sequence[str] | None = None,
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
) -> list[ExperimentSpec]:
    configs = discover_baseline_configs(configs_root, task)
    specs: list[ExperimentSpec] = []
    allowed_bb = set(backbones) if backbones is not None else None
    allowed_attn = set(attentions) if attentions is not None else None
    for cfg in configs:
        backbone, attention = parse_config_stem(cfg.stem)
        if allowed_bb is not None and backbone not in allowed_bb:
            continue
        if allowed_attn is not None and attention not in allowed_attn:
            continue
        for seed in seeds:
            specs.append(
                ExperimentSpec(
                    task=task,
                    backbone=backbone,
                    attention=attention,
                    config_path=str(cfg),
                    seed=int(seed),
                    output_dir=f"runs/{task}/{backbone}_{attention}",
                )
            )
    return specs


def default_backbones_for_task(task: str) -> tuple[str, ...]:
    if task == "classification":
        return CLASSIFICATION_BACKBONES
    if task == "detection":
        return DETECTION_BACKBONES
    if task == "segmentation":
        return SEGMENTATION_BACKBONES
    raise ValueError(f"Unknown task {task}")


class ExperimentRunner:
    """
    Sequential runner over baseline specs.

    ``train_fn(spec) -> metrics dict`` is injected so CLI can bind real trainers
    without importing heavy torch stacks at module import time in dry-run mode.
    """

    def __init__(
        self,
        train_fn: Callable[[ExperimentSpec], dict[str, Any]] | None = None,
        *,
        dry_run: bool = False,
    ) -> None:
        self.train_fn = train_fn
        self.dry_run = dry_run
        self.results: list[ExperimentResult] = []

    def run_one(self, spec: ExperimentSpec) -> ExperimentResult:
        logger.info("Running %s", spec.run_name)
        t0 = time.perf_counter()
        if self.dry_run or self.train_fn is None:
            result = ExperimentResult(
                spec=spec,
                metrics={"dry_run": True},
                elapsed_sec=time.perf_counter() - t0,
                status="dry_run",
            )
            self.results.append(result)
            return result
        try:
            metrics = self.train_fn(spec)
            result = ExperimentResult(
                spec=spec,
                metrics=metrics,
                elapsed_sec=time.perf_counter() - t0,
                status="ok",
            )
        except Exception as exc:  # pragma: no cover - runtime failures
            logger.exception("Experiment failed: %s", spec.run_name)
            result = ExperimentResult(
                spec=spec,
                metrics={},
                elapsed_sec=time.perf_counter() - t0,
                status="error",
                error=str(exc),
            )
        self.results.append(result)
        return result

    def run_all(self, specs: Iterable[ExperimentSpec]) -> list[ExperimentResult]:
        out = []
        for spec in specs:
            out.append(self.run_one(spec))
        return out

    def write_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in self.results]
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        csv_path = path.with_suffix(".csv")
        fieldnames = [
            "task",
            "backbone",
            "attention",
            "seed",
            "status",
            "elapsed_sec",
            "error",
            "config_path",
            "output_dir",
        ]
        # Flatten a few common metrics if present
        metric_keys: set[str] = set()
        for r in rows:
            metric_keys.update(r.get("metrics", {}).keys())
        metric_keys -= {"dry_run"}
        fieldnames.extend(sorted(metric_keys))
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                flat = {k: r.get(k) for k in fieldnames if k in r}
                for mk in metric_keys:
                    flat[mk] = r.get("metrics", {}).get(mk)
                writer.writerow(flat)
        logger.info("Wrote experiment report %s and %s", path, csv_path)


def make_classification_train_fn():
    from maa.training.classification.trainer import ClassificationTrainConfig, ClassificationTrainer

    def _fn(spec: ExperimentSpec) -> dict[str, Any]:
        cfg = ClassificationTrainConfig.from_yaml(spec.config_path)
        cfg.seed = spec.seed
        cfg.output_dir = spec.output_dir
        trainer = ClassificationTrainer(cfg)
        return trainer.train()

    return _fn


def make_detection_train_fn():
    from maa.training.detection.trainer import DetectionTrainConfig, DetectionTrainer

    def _fn(spec: ExperimentSpec) -> dict[str, Any]:
        cfg = DetectionTrainConfig.from_yaml(spec.config_path)
        cfg.seed = spec.seed
        cfg.output_dir = spec.output_dir
        trainer = DetectionTrainer(cfg)
        return trainer.train()

    return _fn


def make_segmentation_train_fn():
    from maa.training.segmentation.trainer import SegmentationTrainConfig, SegmentationTrainer

    def _fn(spec: ExperimentSpec) -> dict[str, Any]:
        cfg = SegmentationTrainConfig.from_yaml(spec.config_path)
        cfg.seed = spec.seed
        cfg.output_dir = spec.output_dir
        trainer = SegmentationTrainer(cfg)
        return trainer.train()

    return _fn


def main(argv: list[str] | None = None) -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Run MAA baseline experiment grid")
    p.add_argument("--task", required=True, choices=["classification", "detection", "segmentation"])
    p.add_argument("--configs-root", default="configs")
    p.add_argument("--backbone", action="append", default=None)
    p.add_argument("--attention", action="append", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=list(EXPERIMENT_SEEDS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", default=None)
    args = p.parse_args(argv)

    specs = build_experiment_grid(
        args.configs_root,
        args.task,
        backbones=args.backbone,
        attentions=args.attention,
        seeds=args.seeds,
    )
    logger.info("Scheduled %d experiments for task=%s", len(specs), args.task)

    train_fn = None
    if not args.dry_run:
        if args.task == "classification":
            train_fn = make_classification_train_fn()
        elif args.task == "detection":
            train_fn = make_detection_train_fn()
        else:
            train_fn = make_segmentation_train_fn()

    runner = ExperimentRunner(train_fn=train_fn, dry_run=args.dry_run)
    runner.run_all(specs)
    report = args.report or f"runs/{args.task}/experiment_report.json"
    runner.write_report(report)


if __name__ == "__main__":
    main()
