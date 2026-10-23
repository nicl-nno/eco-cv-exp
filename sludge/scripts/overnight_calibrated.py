"""Second-night bounded FIT/calibration/outer-validation study; never evaluate test."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import overnight_experiments as night
import pandas as pd
import prepare_grouped_cv as outer
import yaml

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.dataset import build_samples
from sludge_micro.duplicates import near_pairs
from sludge_micro.evaluate import confidence_grid, count_errors, match_all, summarise
from sludge_micro.group_split import group_vectors, split_statistics, train_filter_effect
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json, write_parquet
from sludge_micro.runs import references, run_checks, run_dirs, split_samples
from sludge_micro.runtime import package_digest
from sludge_micro.split import load_split

ROOT = night.ROOT
REPORTS = ROOT / "experiments_updated_2026_10_02/overnight_2026_10_22"
OUTPUTS = ROOT / "outputs_updated_2026_10_02/overnight_2026_10_22"
DEADLINE = datetime.fromisoformat("2026-10-22T08:00:00+03:00")
SEEDS = (20261022, 20261023)
RECIPES = ("control15", "cosine30", "repeat30", "repeat60")
POLICIES = ("fixed_020", "global_f1", "count_guarded")
BASE = night.REPORTS / "queue_configs/cosine30_s20260929.yaml"


def pin(path: Path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}


def verify_pin(record: dict[str, str]) -> None:
    if sha256_file(ROOT / record["path"]) != record["sha256"]:
        raise ValueError(f"Registered input changed: {record['path']}")


def partition_inner(
    training: pd.DataFrame, annotations: pd.DataFrame, classes: tuple[str, ...], seed: int
) -> pd.DataFrame:
    """500 label-balanced group subsets, requiring usable FIT class coverage."""
    if not set(annotations.image_id) <= set(training.image_id):
        raise ValueError("Calibration partition received non-training annotations")
    groups = sorted(training.group.unique())
    if len(groups) < 2:
        raise ValueError("At least two training groups required")
    ignored = set(annotations.loc[annotations.supervision == "ignore", "image_id"])
    vectors = group_vectors(training, annotations, classes)
    usable = group_vectors(training, annotations[~annotations.image_id.isin(ignored)], classes)
    required = usable.sum(axis=0).to_numpy() > 0
    required[0] = False
    values = vectors.to_numpy(float)
    fit_values = usable.to_numpy(float)
    totals = np.maximum(values.sum(axis=0), 1)
    rng = np.random.default_rng(seed)
    best, best_cost = None, float("inf")
    k = max(1, min(len(groups) - 1, round(len(groups) * 0.20)))
    for _ in range(500):
        selected = set(rng.choice(groups, k, replace=False))
        mask = np.array([g in selected for g in vectors.index])
        if np.any((fit_values[~mask].sum(axis=0) == 0) & required):
            continue
        cost = float(np.abs(values[mask].sum(axis=0) / totals - 0.20).sum())
        if cost < best_cost - 1e-12:
            best, best_cost = selected, cost
    if best is None:
        raise ValueError("No calibration partition preserves FIT class support")
    result = training.copy()
    result["split"] = np.where(result.group.isin(best), "validation", "train")
    return result


def check_inner(inner: pd.DataFrame, outer_manifest: pd.DataFrame, hamming: int) -> None:
    expected = outer_manifest[outer_manifest.split == "train"]
    if len(inner) != len(expected) or set(inner.image_id) != set(expected.image_id):
        raise ValueError("Inner membership must be exactly outer training")
    if inner.image_id.duplicated().any() or set(inner.split) != {"train", "validation"}:
        raise ValueError("Invalid inner roles or duplicate IDs")
    indexed = expected.set_index("image_id")
    for key in ("group", "pixel_sha256", "file_sha256", "dhash"):
        if not (inner[key].to_numpy() == inner.image_id.map(indexed[key]).to_numpy()).all():
            raise ValueError(f"Inner identity changed: {key}")
        if (inner.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Inner role leakage: {key}")
    labels = inner.split.tolist()
    if any(labels[i] != labels[j] for i, j, _ in near_pairs(inner.dhash.tolist(), hamming)):
        raise ValueError("Inner near-duplicate leakage")


def verify_registration(registration: dict[str, Any]) -> None:
    if registration["deadline"] != DEADLINE.isoformat() or len(registration["jobs"]) != 24:
        raise ValueError("Registered deadline or budget changed")
    if registration["package_sha256"] != package_digest():
        raise ValueError("Training package changed after registration")
    for record in registration["inputs"]:
        verify_pin(record)
    config = load_config(BASE)
    original = load_split(config)
    for fold in registration["folds"]:
        outer_manifest = pd.read_parquet(ROOT / fold["outer_manifest"])
        outer.check_fold(outer_manifest, original, config.audit.near_duplicate_max_hamming)
        check_inner(
            pd.read_parquet(ROOT / fold["inner_manifest"]),
            outer_manifest,
            config.audit.near_duplicate_max_hamming,
        )
    for job in registration["jobs"]:
        verify_pin(job["config"])


def prepare() -> dict[str, Any]:
    path = REPORTS / "registration.json"
    if path.exists():
        registration = read_json(path)
        verify_registration(registration)
        return registration
    if night.now() >= DEADLINE:
        raise RuntimeError("Night deadline reached")
    config = load_config(BASE)
    original = load_split(config)
    outer.prepare()  # Verifies previously frozen folds, never changes their assignment.
    development = outer.development_only(original)
    _, all_anns = load_prepared(config)
    anns = all_anns[all_anns.image_id.isin(development.image_id)].copy()
    inputs = [
        pin(Path(__file__)),
        pin(ROOT / "scripts/prepare_grouped_cv.py"),
        pin(ROOT / "scripts/overnight_experiments.py"),
        pin(BASE),
        pin(ROOT / "experiments_updated_2026_10_02/split_manifest.parquet"),
        pin(ROOT / "experiments_updated_2026_10_02/test_usage.json"),
        pin(outer.DESTINATION / "fold_registration.json"),
        pin(config.resolve(config.paths.pretrained_weights)),
    ]
    processed = config.resolve(config.paths.processed_dir) / "variants" / config.prepare.variant
    inputs += [pin(processed / name) for name in ("images.parquet", "annotations.parquet")]
    REPORTS.mkdir(parents=True, exist_ok=True)
    protocol = REPORTS / "protocol.md"
    if not protocol.exists():
        protocol.write_bytes((ROOT / "NIGHT_RUN_2026_10_22.md").read_bytes())
    inputs.append(pin(protocol))
    folds = []
    for fold in range(3):
        outer_path = outer.DESTINATION / f"fold_{fold}/split_manifest.parquet"
        manifest = pd.read_parquet(outer_path)
        outer.check_fold(manifest, original, config.audit.near_duplicate_max_hamming)
        training = manifest[manifest.split == "train"].copy()
        training_anns = anns[anns.image_id.isin(training.image_id)]
        inner = partition_inner(training, training_anns, config.classes.names, 20261022 + fold)
        check_inner(inner, manifest, config.audit.near_duplicate_max_hamming)
        directory = REPORTS / f"fold_{fold}"
        inner_path = directory / "split_manifest.parquet"
        if inner_path.exists():
            existing = pd.read_parquet(inner_path)
            pd.testing.assert_frame_equal(existing, inner.reset_index(drop=True))
        else:
            write_parquet(inner_path, inner)
        lock = {
            "manifest": inner_path.name,
            "manifest_sha256": sha256_file(inner_path),
            "data_variant": config.prepare.variant,
            "protocol": {
                "kind": "fit_calibration",
                "outer_fold": fold,
                "original_test_excluded": True,
            },
            "splits": split_statistics(config, inner, training_anns),
            "train_filter": train_filter_effect(config, inner, training_anns),
        }
        night.atomic_json(directory / "split_lock.json", lock)
        night.atomic_json(directory / "test_usage.json", {"disabled": True, "reason": "No test"})
        inputs += [
            pin(outer_path),
            pin(inner_path),
            pin(directory / "split_lock.json"),
            pin(directory / "test_usage.json"),
        ]
        folds.append(
            {
                "fold": fold,
                "outer_manifest": outer_path.relative_to(ROOT).as_posix(),
                "inner_manifest": inner_path.relative_to(ROOT).as_posix(),
                "splits": lock["splits"],
                "train_filter": lock["train_filter"],
            }
        )
    jobs = []
    for seed in SEEDS:
        for fold in range(3):
            for recipe in RECIPES:
                raw = config.model_dump(mode="json", exclude={"root"})
                directory = REPORTS / f"fold_{fold}"
                raw["paths"].update(
                    experiments_dir=directory.relative_to(ROOT).as_posix(),
                    split_dir=directory.relative_to(ROOT).as_posix(),
                    outputs_dir=(OUTPUTS / f"fold_{fold}").relative_to(ROOT).as_posix(),
                )
                raw["runtime"]["seed"] = seed
                raw["train"].update(
                    epochs=60 if recipe == "repeat60" else 30,
                    sampling="repeat_sqrt_020_cap3" if recipe.startswith("repeat") else "none",
                )
                if recipe == "control15":
                    raw["train"].update(epochs=15, lr_schedule="constant", warmup_epochs=0)
                    raw["train"]["augment"].update(rotation90=False, hflip=False, vflip=False)
                raw["protocol"].update(
                    max_runs=8, max_epochs_per_run=60, control_run=None, control_dir=None
                )
                run_id = f"{recipe}_s{seed}"
                cfg_path = REPORTS / "queue_configs" / f"fold_{fold}_{run_id}.yaml"
                cfg_path.parent.mkdir(parents=True, exist_ok=True)
                if (
                    cfg_path.exists()
                    and yaml.safe_load(cfg_path.read_text(encoding="utf-8")) != raw
                ):
                    raise ValueError("Refusing to overwrite configuration")
                if not cfg_path.exists():
                    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
                load_config(cfg_path)
                jobs.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "recipe": recipe,
                        "run_id": run_id,
                        "config": pin(cfg_path),
                    }
                )
    registration = {
        "deadline": DEADLINE.isoformat(),
        "budget": 24,
        "seeds": list(SEEDS),
        "recipes": list(RECIPES),
        "policies": list(POLICIES),
        "jobs": jobs,
        "inputs": inputs,
        "folds": folds,
        "package_sha256": package_digest(),
        "independent_test": False,
        "development_images": len(development),
    }
    verify_registration(registration)
    night.atomic_json(path, registration)
    return registration


def calibrate(
    config: ProjectConfig, predictions: pd.DataFrame, anns: pd.DataFrame, manifest: pd.DataFrame
) -> dict[str, Any]:
    """Only calibration rows may enter; outer outcomes are deliberately not arguments."""
    if set(manifest.split) != {"validation"}:
        raise ValueError("Calibration requires calibration-only manifest")
    ids = manifest.image_id.tolist()
    if len(ids) != len(set(ids)) or not set(predictions.image_id) <= set(ids):
        raise ValueError("Calibration prediction contamination")
    if not set(anns.image_id) <= set(ids):
        raise ValueError("Calibration annotation contamination")
    target, ignore = references(anns)
    det, ref = match_all(predictions, target, ignore, config.evaluation)
    curve = []
    for threshold in confidence_grid(config.evaluation):
        metric = summarise(det, ref, config.classes.names, threshold)
        metric["counting"] = count_errors(det, ref, ids, config.classes.names, threshold)
        curve.append(metric)
    valid = [r for r in curve if r["macro_f1"] is not None]
    global_threshold = (
        max(valid, key=lambda r: (r["macro_f1"], r["confidence"]))["confidence"] if valid else 0.20
    )
    thresholds, details = {}, {}
    groups = manifest.set_index("image_id").group
    for name in config.classes.names:
        part = target[target.class_name == name]
        support, group_support = len(part), int(part.image_id.map(groups).nunique())
        rows = []
        for record in curve:
            c = next(c for c in record["classes"] if c["name"] == name)
            count = next(c for c in record["counting"]["classes"] if c["name"] == name)
            rows.append(
                {
                    "threshold": record["confidence"],
                    "f1": c["f1"] or 0.0,
                    "mae": count["mae_per_frame"],
                }
            )
        best = max(r["f1"] for r in rows)
        reason = "class_count_guarded"
        selected = global_threshold
        if support < 5 or group_support < 3:
            reason = "global_fallback_insufficient_support"
        elif best <= 0:
            reason = "global_fallback_no_true_positives"
        else:
            eligible = [r for r in rows if r["f1"] >= 0.90 * best]
            selected = min(eligible, key=lambda r: (r["mae"], -r["f1"], -r["threshold"]))[
                "threshold"
            ]
        thresholds[name] = selected
        details[name] = {
            "support": support,
            "groups": group_support,
            "reason": reason,
            "best_calibration_f1": best,
            "threshold": selected,
        }
    return {
        "calibration_ids": sorted(ids),
        "class_details": details,
        "policies": {
            "fixed_020": dict.fromkeys(config.classes.names, 0.20),
            "global_f1": dict.fromkeys(config.classes.names, global_threshold),
            "count_guarded": thresholds,
        },
    }


def apply_thresholds(predictions: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    values = predictions.class_name.map(thresholds)
    if values.isna().any():
        raise ValueError("Prediction class has no frozen threshold")
    return predictions[predictions.score >= values].copy()


def measure(
    config: ProjectConfig, predictions: pd.DataFrame, anns: pd.DataFrame, ids: list[str]
) -> dict[str, Any]:
    if len(ids) != len(set(ids)) or not set(predictions.image_id) <= set(ids):
        raise ValueError("Outer prediction contamination or duplicate frame coverage")
    if not set(anns.image_id) <= set(ids):
        raise ValueError("Outer annotation contamination")
    target, ignore = references(anns)
    det, ref = match_all(predictions, target, ignore, config.evaluation)
    result = summarise(det, ref, config.classes.names, 0.0)
    counts = count_errors(det, ref, ids, config.classes.names, 0.0)
    result["confidence"] = counts["confidence"] = None
    return result | {
        "counting": counts,
        "images": len(ids),
        "objects": len(target),
        "confidence_source": "frozen_per_fold_policy_before_outer_inference",
    }


def outer_samples(config: ProjectConfig, fold: dict[str, Any]) -> tuple[list, pd.DataFrame]:
    manifest = pd.read_parquet(ROOT / fold["outer_manifest"])
    ids = set(manifest.loc[manifest.split == "validation", "image_id"])
    images, anns = load_prepared(config)
    images, anns = images[images.image_id.isin(ids)], anns[anns.image_id.isin(ids)]
    return build_samples(config, images, anns), anns


def worker(index: int) -> None:
    from sludge_micro.runs import predict_samples, run_predictor
    from sludge_micro.runtime import apply_runtime
    from sludge_micro.train import run_train

    with night.FileLock(night.OUTPUTS / "gpu.lock"):
        if night.now() >= DEADLINE:
            raise RuntimeError("Night deadline reached")
        registration = read_json(REPORTS / "registration.json")
        verify_registration(registration)
        job = registration["jobs"][index]
        config = load_config(ROOT / job["config"]["path"])
        report_dir, out_dir = run_dirs(config, job["run_id"])
        if not (report_dir / "manifest.json").exists():
            run_train(config, job["run_id"], resume=(out_dir / "resume_state.pt").exists())
        checks = run_checks(config, job["run_id"])
        if any(v is False for v in checks.values()):
            raise ValueError("Completed model/config manifest mismatch")
        _, calibration_anns = split_samples(config, "validation")
        calibration_preds = pd.read_parquet(out_dir / "predictions_validation.parquet")
        manifest = load_split(config)
        selection = calibrate(
            config, calibration_preds, calibration_anns, manifest[manifest.split == "validation"]
        )
        selection["checkpoint_sha256"] = checks["checkpoint_sha256"]
        selection_path = report_dir / "calibration.json"
        if selection_path.exists() and read_json(selection_path) != selection:
            raise ValueError("Refusing to change already frozen calibration")
        night.atomic_json(selection_path, selection)
        apply_runtime(config.runtime)
        samples, anns = outer_samples(config, registration["folds"][job["fold"]])
        predictor = run_predictor(config, job["run_id"], 0.0)
        predictions = predict_samples(predictor, samples)
        write_parquet(out_dir / "predictions_outer_validation.parquet", predictions)
        ids = [s.image_id for s in samples]
        record = {
            "job": job,
            "checks": checks,
            "calibration": pin(selection_path),
            "predictions": pin(out_dir / "predictions_outer_validation.parquet"),
            "policies": {
                p: measure(config, apply_thresholds(predictions, thresholds), anns, ids)
                for p, thresholds in selection["policies"].items()
            },
        }
        night.atomic_json(report_dir / "outer_metrics.json", record)


def summarize(registration: dict[str, Any], queue: list[dict[str, Any]]) -> None:
    records, pooled = [], {}
    for job in queue:
        if job["status"] == "completed":
            config = load_config(ROOT / job["config"]["path"])
            record = read_json(run_dirs(config, job["run_id"])[0] / "outer_metrics.json")
            if record["job"] != next(
                j
                for j in registration["jobs"]
                if j["fold"] == job["fold"] and j["run_id"] == job["run_id"]
            ):
                raise ValueError("Completed result belongs to a different registered job")
            if any(v is False for v in run_checks(config, job["run_id"]).values()):
                raise ValueError("Completed run integrity check failed")
            verify_pin(record["predictions"])
            verify_pin(record["calibration"])
            records.append(record)
    for seed in SEEDS:
        for recipe in RECIPES:
            selected = [
                r for r in records if r["job"]["seed"] == seed and r["job"]["recipe"] == recipe
            ]
            if len(selected) != 3:
                continue
            predictions = {p: [] for p in POLICIES}
            annotations, ids = [], []
            for record in selected:
                verify_pin(record["predictions"])
                verify_pin(record["calibration"])
                job = record["job"]
                config = load_config(ROOT / job["config"]["path"])
                samples, anns = outer_samples(config, registration["folds"][job["fold"]])
                annotations.append(anns)
                ids.extend(s.image_id for s in samples)
                raw = pd.read_parquet(ROOT / record["predictions"]["path"])
                calibration = read_json(ROOT / record["calibration"]["path"])
                for policy in POLICIES:
                    predictions[policy].append(
                        apply_thresholds(raw, calibration["policies"][policy])
                    )
            if len(ids) != registration["development_images"]:
                raise ValueError("Incomplete OOF coverage")
            pooled[f"{recipe}_s{seed}"] = {
                p: measure(
                    config,
                    pd.concat(predictions[p], ignore_index=True),
                    pd.concat(annotations, ignore_index=True),
                    ids,
                )
                for p in POLICIES
            }
    means = {}
    for recipe in RECIPES:
        runs = [pooled[f"{recipe}_s{s}"] for s in SEEDS if f"{recipe}_s{s}" in pooled]
        if len(runs) != len(SEEDS):
            continue
        means[recipe] = {}
        for policy in POLICIES:
            rows = [r[policy] for r in runs]
            classes = []
            for name in config.classes.names:
                cs = [next(c for c in r["classes"] if c["name"] == name) for r in rows]
                counts = [
                    next(c for c in r["counting"]["classes"] if c["name"] == name) for r in rows
                ]
                classes.append(
                    {
                        "name": name,
                        "mean_f1": statistics.mean(c["f1"] for c in cs),
                        "std_f1": statistics.stdev(c["f1"] for c in cs),
                        **{
                            f"mean_{k}": statistics.mean(c[k] for c in counts)
                            for k in (
                                "mae_per_frame",
                                "bias_per_frame",
                                "mae_per_frame_all",
                                "bias_per_frame_all",
                            )
                        },
                        **{
                            f"std_{k}": statistics.stdev(c[k] for c in counts)
                            for k in (
                                "mae_per_frame",
                                "bias_per_frame",
                                "mae_per_frame_all",
                                "bias_per_frame_all",
                            )
                        },
                    }
                )
            means[recipe][policy] = {
                "mean_macro_f1": statistics.mean(r["macro_f1"] for r in rows),
                "std_macro_f1": statistics.stdev(r["macro_f1"] for r in rows),
                "classes": classes,
            }
    eligibility = {}
    if len(means) == len(RECIPES):
        baseline = means["cosine30"]["count_guarded"]
        base_classes = {c["name"]: c for c in baseline["classes"]}
        for recipe in ("repeat30", "repeat60"):
            candidate = means[recipe]["count_guarded"]
            cs = {c["name"]: c for c in candidate["classes"]}
            criteria = {
                "f1_noninferiority": candidate["mean_macro_f1"] >= baseline["mean_macro_f1"] - 0.01,
                "no_class_regression": all(
                    cs[c]["mean_f1"] >= base_classes[c]["mean_f1"] - 0.02 for c in cs
                ),
                **{
                    f"count_gain_{c}": cs[c]["mean_mae_per_frame"]
                    <= 0.9 * base_classes[c]["mean_mae_per_frame"]
                    for c in ("attached_ciliates", "filamentous_bacteria")
                },
            }
            eligibility[recipe] = {"criteria": criteria, "eligible": all(criteria.values())}
    night.atomic_json(
        REPORTS / "summary.json",
        {
            "updated_at": night.now().isoformat(),
            "completed_jobs": len(records),
            "budget": 24,
            "fold_results": records,
            "pooled_per_seed": pooled,
            "means": means,
            "eligibility": eligibility,
            "test_reopened": False,
            "independent_test": False,
        },
    )


def run() -> None:
    if night.now() >= DEADLINE:
        raise RuntimeError("Deadline reached: no new training")
    with night.FileLock(night.OUTPUTS / "queue.lock"):
        with night.FileLock(night.OUTPUTS / "gpu.lock"):
            pass
        registration = prepare()
        path = OUTPUTS / "queue_state.json"
        queue = (
            read_json(path)["queue"]
            if path.exists()
            else [{**j, "status": "pending"} for j in registration["jobs"]]
        )
        if (
            len(queue) != 24
            or [{k: q[k] for k in j} for q, j in zip(queue, registration["jobs"], strict=True)]
            != registration["jobs"]
        ):
            raise ValueError("Queue differs from registration")
        state = {
            "pid": os.getpid(),
            "status": "running",
            "deadline": DEADLINE.isoformat(),
            "queue": queue,
        }
        if any(job["status"] == "failed" for job in queue):
            raise RuntimeError("Failed job requires explicit inspection; no blind restart")
        summarize(registration, queue)
        night.atomic_json(path, state)
        for index, job in enumerate(queue):
            if night.now() >= DEADLINE:
                break
            if job["status"] in {"completed", "failed"}:
                continue
            config = load_config(ROOT / job["config"]["path"])
            if (run_dirs(config, job["run_id"])[0] / "outer_metrics.json").exists():
                job["status"] = "completed"
                night.atomic_json(path, state)
                continue
            job.update(status="running", started_at=night.now().isoformat())
            with (OUTPUTS / f"fold_{job['fold']}_{job['run_id']}.log").open(
                "a", encoding="utf-8"
            ) as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(index)],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "0"},
                )
                job["pid"] = process.pid
                night.atomic_json(path, state)
                try:
                    code = process.wait(timeout=max(1, (DEADLINE - night.now()).total_seconds()))
                    job.update(status="completed" if code == 0 else "failed", exit_code=code)
                except subprocess.TimeoutExpired:
                    night.stop_worker(process)
                    job["status"] = "deadline_stopped"
                finally:
                    job["finished_at"] = night.now().isoformat()
                    night.atomic_json(path, state)
            summarize(registration, queue)
            if job["status"] == "failed":
                state["status"] = "failed_requires_inspection"
                night.atomic_json(path, state)
                return
        state.update(
            status="deadline_reached" if night.now() >= DEADLINE else "queue_finished",
            finished_at=night.now().isoformat(),
        )
        night.atomic_json(path, state)
        summarize(registration, queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    modes.add_argument("--worker", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.prepare:
        print({"jobs": len(prepare()["jobs"]), "deadline": DEADLINE.isoformat()})
    elif args.run:
        run()
    elif args.summarize:
        registration = read_json(REPORTS / "registration.json")
        verify_registration(registration)
        summarize(registration, read_json(OUTPUTS / "queue_state.json")["queue"])
    else:
        worker(args.worker)
