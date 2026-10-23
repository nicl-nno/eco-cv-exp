"""Tabular result aggregation for baseline comparison tables."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PRIMARY_METRICS = {
    "classification": "f1_weighted",
    "detection": "map50",
    "segmentation": "mdice",
}


@dataclass
class RunRow:
    task: str
    backbone: str
    attention: str
    seed: int
    metrics: dict[str, float]


def load_run_metrics(path: str | Path) -> list[RunRow]:
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "results" in raw:
            raw = raw["results"]
        rows = []
        for item in raw:
            rows.append(
                RunRow(
                    task=str(item.get("task", "")),
                    backbone=str(item.get("backbone", "")),
                    attention=str(item.get("attention", "")),
                    seed=int(item.get("seed", 0)),
                    metrics={
                        k: float(v)
                        for k, v in dict(item.get("metrics", {})).items()
                        if isinstance(v, (int, float))
                    },
                )
            )
        return rows
    # CSV
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for item in reader:
            metrics = {}
            for k, v in item.items():
                if k in {"task", "backbone", "attention", "seed", "status", "error", "config_path", "output_dir", "elapsed_sec"}:
                    continue
                try:
                    metrics[k] = float(v)
                except (TypeError, ValueError):
                    continue
            rows.append(
                RunRow(
                    task=item.get("task", ""),
                    backbone=item.get("backbone", ""),
                    attention=item.get("attention", ""),
                    seed=int(float(item.get("seed", 0) or 0)),
                    metrics=metrics,
                )
            )
    return rows


def aggregate_mean_std(rows: Sequence[RunRow], metric: str) -> dict[tuple[str, str], dict[str, float]]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if metric in r.metrics:
            buckets[(r.backbone, r.attention)].append(r.metrics[metric])
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, vals in buckets.items():
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
        out[key] = {"mean": mean, "std": var ** 0.5, "n": float(len(vals))}
    return out


def format_markdown_table(
    rows: Sequence[RunRow],
    *,
    task: str,
    metric: str | None = None,
    attentions: Sequence[str] = ("none", "se", "sa", "cbam", "maa"),
) -> str:
    metric = metric or PRIMARY_METRICS.get(task, "score")
    agg = aggregate_mean_std([r for r in rows if r.task == task or not r.task], metric)
    backbones = sorted({b for b, _ in agg.keys()})
    header = "| Backbone | " + " | ".join(attentions) + " |"
    sep = "|----------|" + "|".join(["------"] * len(attentions)) + "|"
    lines = [header, sep]
    for bb in backbones:
        cells = [bb]
        for attn in attentions:
            stats = agg.get((bb, attn))
            if stats is None:
                cells.append("—")
            else:
                cells.append(f"{stats['mean']:.3f}±{stats['std']:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_per_backbone(rows: Sequence[RunRow], metric: str) -> dict[str, tuple[str, float]]:
    agg = aggregate_mean_std(rows, metric)
    best: dict[str, tuple[str, float]] = {}
    for (bb, attn), stats in agg.items():
        cur = best.get(bb)
        if cur is None or stats["mean"] > cur[1]:
            best[bb] = (attn, stats["mean"])
    return best


def leave_one_out_deltas(
    rows: Sequence[RunRow],
    *,
    metric: str,
    full_attention: str = "maa",
) -> dict[str, float]:
    """
    Expect metrics keyed like ``maa_wo_boundary`` in attention field
    or metric names ``delta_*``. Falls back to attention name prefixes.
    """
    full_vals = [r.metrics[metric] for r in rows if r.attention == full_attention and metric in r.metrics]
    if not full_vals:
        return {}
    full_mean = sum(full_vals) / len(full_vals)
    deltas = {}
    for r in rows:
        if not r.attention.startswith("maa_wo_") and not r.attention.startswith("maa_w/o_"):
            continue
        if metric not in r.metrics:
            continue
        prior = r.attention.replace("maa_wo_", "").replace("maa_w/o_", "")
        deltas[prior] = r.metrics[metric] - full_mean
    return deltas


def write_comparison_report(
    rows: Sequence[RunRow],
    out_dir: str | Path,
    *,
    task: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = PRIMARY_METRICS.get(task, "score")
    md = [
        f"# {task.title()} baseline comparison",
        "",
        f"Primary metric: `{metric}`",
        "",
        format_markdown_table(rows, task=task, metric=metric),
        "",
        "## Best attention per backbone",
        "",
    ]
    for bb, (attn, score) in sorted(best_per_backbone(rows, metric).items()):
        md.append(f"- **{bb}**: {attn} ({score:.4f})")
    path = out_dir / f"{task}_comparison.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def collect_metrics_from_run_dirs(runs_root: str | Path, task: str) -> list[RunRow]:
    """Scan runs/{task}/{backbone}_{attention}/seed_*/metrics.json."""
    root = Path(runs_root) / task
    rows: list[RunRow] = []
    if not root.exists():
        return rows
    for metrics_path in root.glob("*_*/seed_*/metrics.json"):
        parts = metrics_path.parts
        # .../task/backbone_attn/seed_N/metrics.json
        seed_dir = metrics_path.parent.name
        combo = metrics_path.parent.parent.name
        seed = int(seed_dir.replace("seed_", ""))
        backbone, attention = combo.rsplit("_", 1) if "_" in combo else (combo, "unknown")
        # better parse multi-part backbones
        from maa.training.experiments import parse_config_stem

        try:
            backbone, attention = parse_config_stem(combo)
        except Exception:
            pass
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
        rows.append(RunRow(task=task, backbone=backbone, attention=attention, seed=seed, metrics=metrics))
    return rows
