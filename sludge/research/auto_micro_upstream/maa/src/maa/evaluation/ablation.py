"""Leave-one-out MAA prior ablation runners."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from maa.constants import EXPERIMENT_SEEDS, MAA_PRIOR_NAMES
from maa.training.common import load_yaml_config, set_seed

logger = logging.getLogger(__name__)

TaskName = Literal["classification", "detection", "segmentation"]


@dataclass
class AblationRunSpec:
    task: TaskName
    backbone: str
    attention: str = "maa"
    disabled_prior: str | None = None
    seed: int = EXPERIMENT_SEEDS[0]
    output_dir: str = "runs/ablation"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        if self.disabled_prior:
            return f"{self.task}_{self.backbone}_maa_wo_{self.disabled_prior}_seed{self.seed}"
        return f"{self.task}_{self.backbone}_{self.attention}_seed{self.seed}"


def iter_maa_ablation_specs(
    task: TaskName,
    backbone: str,
    *,
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS,
    include_full_maa: bool = True,
) -> list[AblationRunSpec]:
    """Build leave-one-out ablation specs for each MAA prior."""
    specs: list[AblationRunSpec] = []
    if include_full_maa:
        for seed in seeds:
            specs.append(AblationRunSpec(task=task, backbone=backbone, attention="maa", seed=seed))
    for prior in MAA_PRIOR_NAMES:
        for seed in seeds:
            specs.append(
                AblationRunSpec(
                    task=task,
                    backbone=backbone,
                    attention=f"maa_wo_{prior}",
                    disabled_prior=prior,
                    seed=seed,
                )
            )
    return specs


def build_ablation_attention_name(disabled_prior: str | None) -> str:
    if disabled_prior is None:
        return "maa"
    return f"maa_wo_{disabled_prior}"


def run_ablation_grid(
    specs: list[AblationRunSpec],
    train_fn: Callable[[AblationRunSpec], dict[str, Any]],
    eval_fn: Callable[[AblationRunSpec, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Execute a list of ablation specs sequentially.

    ``train_fn`` should train and return a dict containing at least ``checkpoint`` path.
    ``eval_fn`` optionally evaluates the trained checkpoint on the test split.
    """
    results: list[dict[str, Any]] = []
    for spec in specs:
        set_seed(spec.seed)
        logger.info("Ablation run: %s", spec.run_name)
        train_out = train_fn(spec)
        record: dict[str, Any] = {
            "spec": spec.__dict__,
            "train": train_out,
        }
        if eval_fn is not None and "checkpoint" in train_out:
            record["eval"] = eval_fn(spec, train_out["checkpoint"])
        results.append(record)
    return results


def save_ablation_results(results: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")


def summarize_ablation(results: list[dict[str, Any]], metric_key: str) -> dict[str, float]:
    """Aggregate mean metric per disabled prior (None = full MAA)."""
    buckets: dict[str | None, list[float]] = {}
    for row in results:
        spec = row.get("spec", {})
        prior = spec.get("disabled_prior")
        eval_metrics = row.get("eval") or row.get("train", {}).get("metrics", {})
        if metric_key not in eval_metrics:
            continue
        buckets.setdefault(prior, []).append(float(eval_metrics[metric_key]))
    return {str(k): float(sum(v) / len(v)) for k, v in buckets.items() if v}


@dataclass
class AblationConfig:
    task: TaskName = "classification"
    backbone: str = "efficientnetv2"
    output_dir: str = "runs/ablation"
    seeds: tuple[int, ...] = EXPERIMENT_SEEDS
    config_path: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AblationConfig":
        raw = load_yaml_config(path)
        ablation = raw.get("ablation", raw)
        seeds = tuple(ablation.get("seeds", list(EXPERIMENT_SEEDS)))
        return cls(
            task=ablation.get("task", "classification"),
            backbone=ablation.get("backbone", "efficientnetv2"),
            output_dir=str(ablation.get("output_dir", "runs/ablation")),
            seeds=seeds,
            config_path=str(path),
        )
