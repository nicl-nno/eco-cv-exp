"""Six-job, development-only follow-up to the completed overnight experiment queue.

Use --prepare after the first queue finishes, then --run. Never evaluates test.
The shared supervisor/GPU locks prevent overlap with the original night queue.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import overnight_experiments as night
import pandas as pd
import prepare_grouped_cv as folds
import yaml

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.evaluate import count_errors, match_all, summarise
from sludge_micro.reporting import read_json
from sludge_micro.runs import references, run_checks, split_samples

REPORTS = folds.DESTINATION
OUTPUTS = night.ROOT / "outputs_updated_2026_10_02/grouped_cv_2026_10_21"
CONFIDENCE = 0.20


def select_candidate(rows: list[dict[str, Any]]) -> tuple[str | None, dict[str, float]]:
    """Require every registered recipe/seed once before applying the fixed selection rule."""
    expected = {(recipe, seed) for recipe in night.RECIPES for seed in night.SEEDS}
    observed = [(r["recipe"], r["seed"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Selection requires all twelve unique registered recipe/seed pairs")
    if any(not math.isfinite(r["macro_f1"]) for r in rows):
        raise ValueError("Non-finite first-stage metric")
    means = {
        recipe: statistics.mean(r["macro_f1"] for r in rows if r["recipe"] == recipe)
        for recipe in night.RECIPES
    }
    best = min((r for r in means if r != "control15"), key=lambda r: (-means[r], r))
    return (best if means[best] > means["control15"] else None), means


def first_stage_analysis() -> dict[str, Any]:
    """Verify actual completed runs and summarize classes/counts, not just the best seed."""
    state = read_json(night.OUTPUTS / "queue_state.json")
    if state["status"] != "queue_finished" or any(
        q["status"] != "completed" for q in state["queue"]
    ):
        raise ValueError("First-stage queue has not finished successfully; no CV may start")
    rows = []
    for entry in state["queue"]:
        run_id = entry["run_id"]
        if run_id == "release_baseline":
            continue
        config = load_config(night.REPORTS / "queue_configs" / f"{run_id}.yaml")
        checks = run_checks(config, run_id)
        if any(value is False for value in checks.values()):
            raise ValueError(f"First-stage immutable manifest check failed: {run_id}")
        if checks["split_manifest_sha256"] != folds.EXPECTED_SOURCE_SHA:
            raise ValueError("First-stage runs do not share the registered split")
        metric = read_json(night.REPORTS / run_id / "metrics.json")
        counting = read_json(night.REPORTS / run_id / "validation_counting.json")
        rows.append(
            {
                "run_id": run_id,
                "recipe": entry["recipe"],
                "seed": entry["seed"],
                "macro_f1": metric["macro_f1"],
                "confidence": metric["confidence"],
                "classes": metric["classes"],
                "counting": counting,
                "checks": checks,
            }
        )
    selected, means = select_candidate(rows)
    recipes = {}
    for recipe in night.RECIPES:
        group = [r for r in rows if r["recipe"] == recipe]
        names = [c["name"] for c in group[0]["classes"]]
        classes = []
        for name in names:
            metrics = [next(c for c in r["classes"] if c["name"] == name) for r in group]
            counts = [next(c for c in r["counting"]["classes"] if c["name"] == name) for r in group]
            classes.append(
                {
                    "name": name,
                    "support": metrics[0]["support"],
                    "mean_f1": statistics.mean(m["f1"] for m in metrics),
                    "std_f1": statistics.stdev(m["f1"] for m in metrics),
                    "mean_count_mae": statistics.mean(c["mae_per_frame"] for c in counts),
                    "mean_count_bias": statistics.mean(c["bias_per_frame"] for c in counts),
                    "mean_count_mae_all": statistics.mean(c["mae_per_frame_all"] for c in counts),
                }
            )
        recipes[recipe] = {
            "mean_macro_f1": means[recipe],
            "std_macro_f1": statistics.stdev(r["macro_f1"] for r in group),
            "classes": classes,
        }
    paired = []
    if selected:
        for seed in night.SEEDS:
            control = next(r for r in rows if r["seed"] == seed and r["recipe"] == "control15")
            candidate = next(r for r in rows if r["seed"] == seed and r["recipe"] == selected)
            paired.append(
                {"seed": seed, "delta_macro_f1": candidate["macro_f1"] - control["macro_f1"]}
            )
    return {
        "selected": selected,
        "recipes": recipes,
        "runs": rows,
        "paired_deltas": paired,
        "warning": "Development validation, with per-run threshold selection; three seeds "
        "measure training variability, not independent generalization uncertainty.",
    }


def prepare() -> dict[str, Any]:
    """Freeze the selected recipe and six configs only after successful first-stage completion."""
    registration = folds.prepare()
    analysis = first_stage_analysis()
    selected = analysis["selected"]
    entries = []
    if selected:
        for fold in range(3):
            for recipe in ("control15", selected):
                raw = load_config(
                    night.REPORTS / "queue_configs" / f"{recipe}_s{night.SEEDS[0]}.yaml"
                ).model_dump(mode="json", exclude={"root"})
                report_dir = REPORTS / f"fold_{fold}"
                out_dir = OUTPUTS / f"fold_{fold}"
                raw["paths"].update(
                    experiments_dir=report_dir.relative_to(night.ROOT).as_posix(),
                    split_dir=report_dir.relative_to(night.ROOT).as_posix(),
                    outputs_dir=out_dir.relative_to(night.ROOT).as_posix(),
                )
                raw["runtime"]["seed"] = 20261021
                raw["protocol"].update(max_runs=2, control_run="control15", control_dir=None)
                path = REPORTS / "queue_configs" / f"fold_{fold}_{recipe}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if yaml.safe_load(path.read_text(encoding="utf-8")) != raw:
                        raise ValueError("Refusing to change an existing CV configuration")
                else:
                    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
                load_config(path)
                entries.append(
                    {
                        "fold": fold,
                        "recipe": recipe,
                        "run_id": recipe,
                        "config": path.relative_to(night.ROOT).as_posix(),
                        "config_sha256": sha256_file(path),
                    }
                )
    result = {
        "selected": selected,
        "fixed_confidence": CONFIDENCE,
        "seed": 20261021,
        "fold_registration_sha256": sha256_file(REPORTS / "fold_registration.json"),
        "development_images": registration["development_images"],
        "jobs": entries,
        "deadline": night.DEADLINE.isoformat(),
        "independent_external_test": False,
        "first_stage_means": {r: v["mean_macro_f1"] for r, v in analysis["recipes"].items()},
    }
    path = REPORTS / "selection.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Registered selection changed")
    night.atomic_json(path, result)
    night.atomic_json(REPORTS / "first_stage_analysis.json", analysis)
    return result


def job_config(job: dict[str, Any]) -> ProjectConfig:
    path = night.ROOT / job["config"]
    if sha256_file(path) != job["config_sha256"]:
        raise ValueError("Registered job config changed")
    config = load_config(path)
    if not (config.resolve(config.paths.experiments_dir) / "test_usage.json").exists():
        raise ValueError("CV test guard is missing")
    return config


def fixed_metrics(
    config: ProjectConfig, predictions: pd.DataFrame, anns: pd.DataFrame, image_ids: list[str]
) -> dict[str, Any]:
    if len(image_ids) != len(set(image_ids)) or not set(predictions.image_id) <= set(image_ids):
        raise ValueError("Duplicate validation IDs or out-of-fold prediction contamination")
    target, ignore = references(anns)
    if not set(anns.image_id) <= set(image_ids):
        raise ValueError("Out-of-fold annotation contamination")
    det, ref = match_all(predictions, target, ignore, config.evaluation)
    return summarise(det, ref, config.classes.names, CONFIDENCE) | {
        "images": len(image_ids),
        "objects": len(target),
        "confidence_source": "preregistered_fixed_0.20",
        "counting": count_errors(det, ref, image_ids, config.classes.names, CONFIDENCE),
    }


def worker(index: int) -> None:
    from sludge_micro.train import run_train

    with night.FileLock(night.OUTPUTS / "gpu.lock"):
        if night.now() >= night.DEADLINE:
            raise RuntimeError("Night deadline reached")
        selection = read_json(REPORTS / "selection.json")
        if sha256_file(REPORTS / "fold_registration.json") != selection["fold_registration_sha256"]:
            raise ValueError("Fold registration changed")
        job = selection["jobs"][index]
        config = job_config(job)
        run_id = job["run_id"]
        out_dir = config.resolve(config.paths.outputs_dir) / run_id
        run_train(config, run_id, resume=(out_dir / "resume_state.pt").exists())
        samples, annotations = split_samples(config, "validation")
        predictions = pd.read_parquet(out_dir / "predictions_validation.parquet")
        record = fixed_metrics(config, predictions, annotations, [s.image_id for s in samples])
        record.update(fold=job["fold"], recipe=job["recipe"], checks=run_checks(config, run_id))
        night.atomic_json(
            config.resolve(config.paths.experiments_dir) / run_id / "fixed_metrics.json", record
        )


def report(selection: dict[str, Any], queue: list[dict[str, Any]]) -> None:
    complete = [q for q in queue if q["status"] == "completed"]
    records = []
    for job in complete:
        config = job_config(job)
        records.append(
            read_json(
                config.resolve(config.paths.experiments_dir) / job["run_id"] / "fixed_metrics.json"
            )
        )
    pooled = {}
    if len(complete) == 6:
        for recipe in ("control15", selection["selected"]):
            predictions, annotations, image_ids = [], [], []
            for job in complete:
                if job["recipe"] != recipe:
                    continue
                config = job_config(job)
                checks = run_checks(config, job["run_id"])
                if any(value is False for value in checks.values()):
                    raise ValueError("Completed CV manifest check failed")
                samples, anns = split_samples(config, "validation")
                image_ids.extend(s.image_id for s in samples)
                annotations.append(anns)
                predictions.append(
                    pd.read_parquet(
                        config.resolve(config.paths.outputs_dir)
                        / job["run_id"]
                        / "predictions_validation.parquet"
                    )
                )
            if len(image_ids) != selection["development_images"]:
                raise ValueError("Incomplete out-of-fold coverage")
            pooled[recipe] = fixed_metrics(
                config,
                pd.concat(predictions, ignore_index=True),
                pd.concat(annotations, ignore_index=True),
                image_ids,
            )
    night.atomic_json(
        REPORTS / "summary.json",
        {
            "updated_at": night.now().isoformat(),
            "selected": selection["selected"],
            "completed_jobs": len(complete),
            "budget": 6,
            "fold_metrics": records,
            "pooled": pooled,
            "delta_macro_f1": (
                pooled[selection["selected"]]["macro_f1"] - pooled["control15"]["macro_f1"]
                if pooled
                else None
            ),
            "confidence": CONFIDENCE,
            "test_reopened": False,
            "warning": "Development check, not nested CV or an independent external test.",
        },
    )


def run() -> None:
    if night.now() >= night.DEADLINE:
        raise RuntimeError("Night deadline reached; no new training")
    # Original supervisor holds this lock until all jobs are finished, even between workers.
    with night.FileLock(night.OUTPUTS / "queue.lock"):
        with night.FileLock(night.OUTPUTS / "gpu.lock"):
            pass
        selection = prepare()
        if not selection["selected"]:
            print("No candidate exceeded control; CV not started.")
            return
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS / "queue_state.json"
        queue = (
            read_json(path)["queue"]
            if path.exists()
            else [{**j, "status": "pending"} for j in selection["jobs"]]
        )
        if [
            {k: q[k] for k in j} for q, j in zip(queue, selection["jobs"], strict=True)
        ] != selection["jobs"]:
            raise ValueError("CV queue differs from registered jobs")
        state = {
            "pid": os.getpid(),
            "status": "running",
            "deadline": night.DEADLINE.isoformat(),
            "queue": queue,
        }
        night.atomic_json(path, state)
        for index, entry in enumerate(queue):
            if night.now() >= night.DEADLINE:
                break
            if entry["status"] in {"completed", "failed"}:
                continue
            config = job_config(entry)
            completion = (
                config.resolve(config.paths.experiments_dir)
                / entry["run_id"]
                / "fixed_metrics.json"
            )
            if completion.exists():
                entry["status"] = "completed"
                continue
            entry.update(status="running", started_at=night.now().isoformat())
            log_path = OUTPUTS / f"fold_{entry['fold']}_{entry['recipe']}.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(index)],
                    cwd=night.ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1"},
                )
                entry["pid"] = process.pid
                night.atomic_json(path, state)
                try:
                    result = process.wait(
                        timeout=max(1, (night.DEADLINE - night.now()).total_seconds())
                    )
                    entry.update(status="completed" if result == 0 else "failed", exit_code=result)
                except subprocess.TimeoutExpired:
                    night.stop_worker(process)
                    entry["status"] = "deadline_stopped"
                finally:
                    entry["finished_at"] = night.now().isoformat()
                    night.atomic_json(path, state)
            report(selection, queue)
        state["status"] = "deadline_reached" if night.now() >= night.DEADLINE else "queue_finished"
        night.atomic_json(path, state)
        report(selection, queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.prepare:
        print(prepare()["selected"])
    elif args.run:
        run()
    else:
        worker(args.worker)
