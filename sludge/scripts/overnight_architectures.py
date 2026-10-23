"""Third-night registered cascade/specialists study; existing controls, no test."""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import architecture_components as components
import overnight_calibrated as old
import overnight_experiments as night
import pandas as pd
import yaml

from sludge_micro.config import load_config
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json
from sludge_micro.runs import split_samples
from sludge_micro.runtime import package_digest
from sludge_micro.split import load_split

ROOT = night.ROOT
REPORTS = ROOT / "experiments_updated_2026_10_02/overnight_2026_10_23"
OUTPUTS = ROOT / "outputs_updated_2026_10_02/overnight_2026_10_23"
DEADLINE = datetime.fromisoformat("2026-10-23T08:00:00+03:00")
METHODS = ("cosine30", "cascade", "specialists", "specialists_union")
SEEDS = old.SEEDS
POLICIES = old.POLICIES


def job_dir(job, reports=False):
    return (REPORTS if reports else OUTPUTS) / f"fold_{job['fold']}" / job["run_id"]


def verify(reg):
    if (
        reg["deadline"] != DEADLINE.isoformat()
        or len(reg["jobs"]) != 12
        or reg["component_fits"] != 24
    ):
        raise ValueError("Registered deadline/budget mismatch")
    expected = {(s, f, m) for s in SEEDS for f in range(3) for m in ("cascade", "specialists")}
    if {(j["seed"], j["fold"], j["recipe"]) for j in reg["jobs"]} != expected:
        raise ValueError("Registered jobs changed")
    if reg["package_sha256"] != package_digest():
        raise ValueError("Installed package changed")
    for record in reg["inputs"]:
        old.verify_pin(record)
    old.verify_registration(read_json(old.REPORTS / "registration.json"))
    for job in reg["jobs"]:
        old.verify_pin(job["config"])
        for record in job["components"].values():
            old.verify_pin(record)
    for control in reg["controls"]:
        old.verify_pin(control["config"])
        for record in control["artifacts"]:
            old.verify_pin(record)


def prepare():
    path = REPORTS / "registration.json"
    if path.exists():
        reg = read_json(path)
        verify(reg)
        return reg
    if night.now() >= DEADLINE:
        raise RuntimeError("Deadline: no new registration/training")
    prior = read_json(old.REPORTS / "registration.json")
    old.verify_registration(prior)
    REPORTS.mkdir(parents=True, exist_ok=True)
    protocol = REPORTS / "protocol.md"
    source = ROOT / "NIGHT_RUN_2026_10_23.md"
    if protocol.exists() and protocol.read_bytes() != source.read_bytes():
        raise ValueError("Protocol already exists with different content")
    if not protocol.exists():
        protocol.write_bytes(source.read_bytes())
    inputs = [
        old.pin(p)
        for p in (
            source,
            protocol,
            Path(__file__).resolve(),
            ROOT / "scripts/architecture_components.py",
            ROOT / "scripts/overnight_calibrated.py",
            old.REPORTS / "registration.json",
            ROOT / "tests/test_overnight_architectures.py",
        )
    ]
    base_config = load_config(old.BASE)
    development = old.outer.development_only(load_split(base_config))
    images, _ = load_prepared(base_config)
    images = images[images.image_id.isin(development.image_id)]
    if len(images) != prior["development_images"]:
        raise ValueError("Development image file coverage mismatch")
    for row in images.itertuples():
        image_pin = old.pin(base_config.resolve(row.path))
        if image_pin["sha256"] != row.file_sha256:
            raise ValueError("Development image no longer matches frozen metadata")
        inputs.append(image_pin)
    controls, jobs = [], []
    for seed in SEEDS:
        for fold in range(3):
            control = next(
                j
                for j in prior["jobs"]
                if j["seed"] == seed and j["fold"] == fold and j["recipe"] == "cosine30"
            )
            config = load_config(ROOT / control["config"]["path"])
            report, output = old.run_dirs(config, control["run_id"])
            if any(v is False for v in old.run_checks(config, control["run_id"]).values()):
                raise ValueError("Control failed integrity checks")
            artifacts = [
                old.pin(p)
                for p in (
                    report / "manifest.json",
                    report / "outer_metrics.json",
                    report / "calibration.json",
                    output / "model.pt",
                    output / "predictions_validation.parquet",
                    output / "predictions_outer_validation.parquet",
                )
            ]
            controls.append({**control, "artifacts": artifacts})
            for method in ("cascade", "specialists"):
                configs = {}
                for component in (
                    ("agnostic",) if method == "cascade" else ("filamentous", "others")
                ):
                    raw = config.model_dump(mode="json", exclude={"root"})
                    labels = components.mapping(config.classes.names, component)
                    target = []
                    if component == "agnostic":
                        target = [
                            {
                                **raw["classes"]["target"][0],
                                "label": 1,
                                "name": "microorganism",
                                "title": "Target microorganism",
                            }
                        ]
                    else:
                        target = [
                            {**t, "label": labels[t["label"]]}
                            for t in raw["classes"]["target"]
                            if t["label"] in labels
                        ]
                    raw["classes"]["target"] = target
                    raw["paths"].update(
                        outputs_dir=OUTPUTS.relative_to(ROOT).as_posix(),
                        experiments_dir=REPORTS.relative_to(ROOT).as_posix(),
                    )
                    cfg = (
                        REPORTS / "queue_configs" / f"fold_{fold}_{method}_{component}_s{seed}.yaml"
                    )
                    cfg.parent.mkdir(parents=True, exist_ok=True)
                    if cfg.exists() and yaml.safe_load(cfg.read_text(encoding="utf-8")) != raw:
                        raise ValueError("Refusing to overwrite component config")
                    if not cfg.exists():
                        cfg.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
                    load_config(cfg)
                    configs[component] = old.pin(cfg)
                jobs.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "recipe": method,
                        "run_id": f"{method}_s{seed}",
                        "config": control["config"],
                        "components": configs,
                    }
                )
    reg = {
        "created_at": night.now().isoformat(),
        "deadline": DEADLINE.isoformat(),
        "component_fits": 24,
        "jobs": jobs,
        "controls": controls,
        "folds": prior["folds"],
        "inputs": inputs,
        "package_sha256": package_digest(),
        "development_images": prior["development_images"],
        "independent_test": False,
        "primary": "count_guarded",
        "seeds": list(SEEDS),
    }
    verify(reg)
    night.atomic_json(path, reg)
    return reg


def freeze_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to change frozen artifact: {path}")
    else:
        night.atomic_json(path, value)


def require_calibration(report, methods):
    paths = [report / f"calibration_{method}.json" for method in methods]
    if not all(p.is_file() for p in paths):
        raise ValueError("All policies must be frozen before outer inference")
    return [old.pin(p) for p in paths]


def worker(index):
    with night.FileLock(night.OUTPUTS / "gpu.lock"):
        if night.now() >= DEADLINE:
            raise RuntimeError("Deadline: no new worker")
        reg = read_json(REPORTS / "registration.json")
        verify(reg)
        job = reg["jobs"][index]
        output, report = job_dir(job), job_dir(job, True)
        output.mkdir(parents=True, exist_ok=True)
        report.mkdir(parents=True, exist_ok=True)
        config = load_config(ROOT / job["config"]["path"])
        samples, _ = split_samples(config, "train")
        calibration_samples, calibration_anns = split_samples(config, "validation")
        manifest = load_split(config)
        if not {s.image_id for s in samples} <= set(
            manifest.loc[manifest.split == "train", "image_id"]
        ):
            raise ValueError("FIT contamination")
        component_configs = {
            name: load_config(ROOT / p["path"]) for name, p in job["components"].items()
        }
        for name, cfg in component_configs.items():
            remapped = components.remap_samples(
                samples, components.mapping(config.classes.names, name)
            )
            components.fit_detector(cfg, remapped, output / name, DEADLINE)
        if job["recipe"] == "cascade":
            fit_preds = components.detector_predictions(
                component_configs["agnostic"],
                output / "agnostic",
                samples,
                output / "proposals_fit.parquet",
            )
            crops = components.crop_manifest(samples, fit_preds, job["seed"])
            crop_path = output / "classifier/crops.parquet"
            if crop_path.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(crop_path), crops)
            else:
                components.atomic_parquet(crop_path, crops)
            freeze_json(
                report / "crop_provenance.json",
                {
                    "manifest": old.pin(crop_path),
                    "fit_predictions": old.pin(output / "proposals_fit.parquet"),
                    "fit_ids": sorted(s.image_id for s in samples),
                    "inner_manifest": old.pin(ROOT / reg["folds"][job["fold"]]["inner_manifest"]),
                    "negative_limitations": (
                        "in-sample mining; annotation-relative background, "
                        "potentially incomplete labels"
                    ),
                },
            )
            components.fit_classifier(config, samples, crops, output / "classifier", DEADLINE)
        methods = (
            ["cascade"] if job["recipe"] == "cascade" else ["specialists", "specialists_union"]
        )

        def inference(stage, stage_samples):
            if job["recipe"] == "cascade":
                proposals = components.detector_predictions(
                    component_configs["agnostic"],
                    output / "agnostic",
                    stage_samples,
                    output / f"proposals_{stage}.parquet",
                )
                result = components.classify_proposals(
                    config,
                    output / "classifier",
                    stage_samples,
                    proposals,
                    output / f"predictions_cascade_{stage}.parquet",
                )
                return {"cascade": result}
            first = components.detector_predictions(
                component_configs["filamentous"],
                output / "filamentous",
                stage_samples,
                output / f"filamentous_{stage}.parquet",
            )
            second = components.detector_predictions(
                component_configs["others"],
                output / "others",
                stage_samples,
                output / f"others_{stage}.parquet",
            )
            merged = {
                m: components.merge_specialists(
                    first, second, suppress=(m == "specialists"), return_counts=True
                )
                for m in methods
            }
            results = {m: pair[0] for m, pair in merged.items()}
            for m, frame in results.items():
                components.atomic_parquet(output / f"predictions_{m}_{stage}.parquet", frame)
            freeze_json(
                report / f"merge_{stage}.json",
                {m: pair[1] for m, pair in merged.items()},
            )
            return results

        cal_predictions = inference("calibration", calibration_samples)
        completions = [
            old.pin(p / "complete.json")
            for p in output.iterdir()
            if p.is_dir() and (p / "complete.json").exists()
        ]
        for method in methods:
            selection = old.calibrate(
                config,
                cal_predictions[method],
                calibration_anns,
                manifest[manifest.split == "validation"],
            )
            selection["component_completions"] = completions
            freeze_json(report / f"calibration_{method}.json", selection)
        calibration_pins = require_calibration(report, methods)
        freeze_json(
            report / "outer_authorization.json",
            {"calibrations": calibration_pins, "original_test_excluded": True},
        )
        outer_samples, anns = old.outer_samples(config, reg["folds"][job["fold"]])
        results = inference("outer_validation", outer_samples)
        records = []
        for method in methods:
            selection = read_json(report / f"calibration_{method}.json")
            record = {
                "job": job,
                "method": method,
                "predictions": old.pin(output / f"predictions_{method}_outer_validation.parquet"),
                "calibration": old.pin(report / f"calibration_{method}.json"),
                "component_completions": completions,
                "policies": {
                    p: old.measure(
                        config,
                        old.apply_thresholds(results[method], selection["policies"][p]),
                        anns,
                        [s.image_id for s in outer_samples],
                    )
                    for p in POLICIES
                },
                "diagnostics": components.diagnostics(config, results[method], anns),
            }
            if method == "cascade":
                record["agnostic_proposals_diagnostics"] = components.diagnostics(
                    config, pd.read_parquet(output / "proposals_outer_validation.parquet"), anns
                )
                record["agnostic_proposals_diagnostics"]["matched_class_accuracy"] = None
                record["agnostic_proposals_diagnostics"]["matched_class_correct"] = None
            freeze_json(report / f"metrics_{method}.json", record)
            records.append(old.pin(report / f"metrics_{method}.json"))
        freeze_json(
            report / "complete.json",
            {
                "job": job,
                "records": records,
                "component_completions": completions,
                "diagnostic_artifacts": [old.pin(p) for p in sorted(output.glob("*.timing.json"))]
                + [old.pin(p) for p in sorted(report.glob("merge_*.json"))],
            },
        )


def verify_completed(job):
    final = read_json(job_dir(job, True) / "complete.json")
    if final["job"] != job:
        raise ValueError("Completed job identity mismatch")
    for pin in final["component_completions"]:
        old.verify_pin(pin)
        for artifact in read_json(ROOT / pin["path"])["artifacts"]:
            old.verify_pin(artifact)
    records = []
    for pin in final["records"]:
        old.verify_pin(pin)
        record = read_json(ROOT / pin["path"])
        old.verify_pin(record["predictions"])
        old.verify_pin(record["calibration"])
        records.append(record)
    for pin in final.get("diagnostic_artifacts", []):
        old.verify_pin(pin)
    return records


def seed_means(pooled, names):
    result = {}
    for method in METHODS:
        if not all(f"{method}_s{s}" in pooled for s in SEEDS):
            continue
        result[method] = {}
        for policy in POLICIES:
            rows = [pooled[f"{method}_s{s}"][policy] for s in SEEDS]
            classes = []
            for name in names:
                det = [next(c for c in row["classes"] if c["name"] == name) for row in rows]
                counts = [
                    next(c for c in row["counting"]["classes"] if c["name"] == name) for row in rows
                ]
                item = {"name": name}
                for key in ("f1", "precision", "recall"):
                    values = [c[key] for c in det]
                    item[f"mean_{key}"] = statistics.mean(values) if None not in values else None
                    item[f"std_{key}"] = statistics.stdev(values) if None not in values else None
                for key in (
                    "mae_per_frame",
                    "bias_per_frame",
                    "mae_per_frame_all",
                    "bias_per_frame_all",
                ):
                    item[f"mean_{key}"] = statistics.mean(c[key] for c in counts)
                    item[f"std_{key}"] = statistics.stdev(c[key] for c in counts)
                classes.append(item)
            result[method][policy] = {
                "mean_macro_f1": statistics.mean(r["macro_f1"] for r in rows),
                "std_macro_f1": statistics.stdev(r["macro_f1"] for r in rows),
                "classes": classes,
            }
    return result


def eligibility(means):
    result = {}
    baseline = means["cosine30"]["count_guarded"]
    base = {c["name"]: c for c in baseline["classes"]}
    for method in ("cascade", "specialists"):
        if method not in means:
            continue
        candidate = means[method]["count_guarded"]
        cs = {c["name"]: c for c in candidate["classes"]}
        criteria = {
            "f1_noninferiority": candidate["mean_macro_f1"] >= baseline["mean_macro_f1"] - 0.01,
            "no_class_regression": all(
                cs[c]["mean_f1"] is not None
                and base[c]["mean_f1"] is not None
                and cs[c]["mean_f1"] >= base[c]["mean_f1"] - 0.02
                for c in cs
            ),
            **{
                f"count_gain_{c}": cs[c]["mean_mae_per_frame"]
                <= 0.9 * base[c]["mean_mae_per_frame"]
                for c in ("attached_ciliates", "filamentous_bacteria")
            },
        }
        result[method] = {"criteria": criteria, "eligible": all(criteria.values())}
    return result


def summarize(reg, queue):
    records = []
    for control in reg["controls"]:
        config = load_config(ROOT / control["config"]["path"])
        report, _ = old.run_dirs(config, control["run_id"])
        record = read_json(report / "outer_metrics.json")
        record["method"] = "cosine30"
        records.append(record)
    completed = 0
    for job, state in zip(reg["jobs"], queue, strict=True):
        if state["status"] == "completed":
            records.extend(verify_completed(job))
            completed += 1
    pooled, diagnostic_totals = {}, {}
    for seed in SEEDS:
        for method in METHODS:
            selected = [r for r in records if r["job"]["seed"] == seed and r["method"] == method]
            if len(selected) != 3:
                continue
            if {r["job"]["fold"] for r in selected} != {0, 1, 2}:
                raise ValueError("Duplicate/missing OOF fold")
            predictions = {p: [] for p in POLICIES}
            annotations, ids, diagnostics = [], [], []
            for record in selected:
                old.verify_pin(record["predictions"])
                old.verify_pin(record["calibration"])
                job = record["job"]
                config = load_config(ROOT / job["config"]["path"])
                samples, anns = old.outer_samples(config, reg["folds"][job["fold"]])
                ids.extend(s.image_id for s in samples)
                annotations.append(anns)
                raw = pd.read_parquet(ROOT / record["predictions"]["path"])
                calibration = read_json(ROOT / record["calibration"]["path"])
                diagnostics.append(components.diagnostics(config, raw, anns))
                for p in POLICIES:
                    predictions[p].append(old.apply_thresholds(raw, calibration["policies"][p]))
            if len(ids) != reg["development_images"] or len(ids) != len(set(ids)):
                raise ValueError("Incomplete/duplicate OOF image coverage")
            pooled[f"{method}_s{seed}"] = {
                p: old.measure(
                    config,
                    pd.concat(predictions[p], ignore_index=True),
                    pd.concat(annotations, ignore_index=True),
                    ids,
                )
                for p in POLICIES
            }
            totals = {
                k: sum(d[k] for d in diagnostics)
                for k in (
                    "localization_tp",
                    "localization_fp",
                    "localization_fn",
                    "ignored",
                    "duplicate_fp",
                    "matched_class_correct",
                )
            }
            tp, fp, fn = (
                totals[k] for k in ("localization_tp", "localization_fp", "localization_fn")
            )
            totals.update(
                localization_precision=tp / (tp + fp) if tp + fp else None,
                localization_recall=tp / (tp + fn) if tp + fn else None,
                matched_class_accuracy=totals["matched_class_correct"] / tp if tp else None,
            )
            diagnostic_totals[f"{method}_s{seed}"] = totals
    means = seed_means(pooled, config.classes.names)
    night.atomic_json(
        REPORTS / "summary.json",
        {
            "updated_at": night.now().isoformat(),
            "completed_pipeline_jobs": completed,
            "budget_pipeline_jobs": 12,
            "component_fit_budget": 24,
            "reused_control_jobs": 6,
            "fold_results": records,
            "pooled_per_seed": pooled,
            "means": means,
            "eligibility": eligibility(means),
            "localization_diagnostics_per_seed": diagnostic_totals,
            "test_reopened": False,
            "independent_test": False,
            "primary": "count_guarded",
        },
    )


def validate_queue(reg, queue):
    if (
        len(queue) != 12
        or [{k: q[k] for k in j} for q, j in zip(queue, reg["jobs"], strict=True)] != reg["jobs"]
    ):
        raise ValueError("Queue differs from frozen registration")
    if any(j["status"] == "failed" for j in queue):
        raise RuntimeError("Failed job requires inspection; no blind retry")


def run():
    if night.now() >= DEADLINE:
        raise RuntimeError("Deadline reached: no new training")
    with night.FileLock(night.OUTPUTS / "queue.lock"):
        with night.FileLock(night.OUTPUTS / "gpu.lock"):
            pass
        reg = prepare()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS / "queue_state.json"
        queue = (
            read_json(path)["queue"]
            if path.exists()
            else [{**j, "status": "pending"} for j in reg["jobs"]]
        )
        validate_queue(reg, queue)
        for job, state in zip(reg["jobs"], queue, strict=True):
            if (job_dir(job, True) / "complete.json").exists():
                verify_completed(job)
                state["status"] = "completed"
        state = {
            "pid": os.getpid(),
            "status": "running",
            "deadline": DEADLINE.isoformat(),
            "queue": queue,
        }
        night.atomic_json(path, state)
        summarize(reg, queue)
        for index, job in enumerate(queue):
            if night.now() >= DEADLINE:
                break
            if job["status"] == "completed":
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
            summarize(reg, queue)
            if job["status"] == "failed":
                state["status"] = "failed_requires_inspection"
                night.atomic_json(path, state)
                return
        state.update(
            status="deadline_reached" if night.now() >= DEADLINE else "queue_finished",
            finished_at=night.now().isoformat(),
        )
        night.atomic_json(path, state)
        summarize(reg, queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--summarize", action="store_true")
    modes.add_argument("--worker", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.prepare:
        reg = prepare()
        print({"jobs": len(reg["jobs"]), "component_fits": 24, "deadline": reg["deadline"]})
    elif args.run:
        run()
    elif args.summarize:
        reg = read_json(REPORTS / "registration.json")
        verify(reg)
        queue_path = OUTPUTS / "queue_state.json"
        queue = (
            read_json(queue_path)["queue"]
            if queue_path.exists()
            else [{**j, "status": "pending"} for j in reg["jobs"]]
        )
        summarize(reg, queue)
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
