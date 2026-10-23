"""Bounded selected-class specialist study; separate registration, never test."""

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
import overnight_architectures as previous
import overnight_calibrated as old
import overnight_experiments as night
import pandas as pd
import yaml

from sludge_micro.config import load_config
from sludge_micro.dataset import sample_weights
from sludge_micro.reporting import read_json
from sludge_micro.runs import split_samples
from sludge_micro.runtime import package_digest
from sludge_micro.split import load_split

ROOT = night.ROOT
REPORTS = ROOT / "experiments_updated_2026_10_02/overnight_2026_10_23_hybrids"
OUTPUTS = ROOT / "outputs_updated_2026_10_02/overnight_2026_10_23_hybrids"
DEADLINE = datetime.fromisoformat("2026-10-23T08:00:00+03:00")
SEEDS = old.SEEDS
POLICIES = old.POLICIES
ATTACHED = "attached_ciliates"
FILAMENT = "filamentous_bacteria"
RARE = ("free_swimming_ciliates", "nematoda", "gastrotrichs")
GROUPS = {
    "attached": (ATTACHED,),
    "priority": (ATTACHED, FILAMENT),
    "rare": RARE,
    "priority_repeat": (ATTACHED, FILAMENT),
    "rare_repeat": RARE,
}
REPLACEMENTS = {
    "h_filamentous": {FILAMENT: "filamentous"},
    "h_attached": {ATTACHED: "attached"},
    "h_priority": dict.fromkeys((ATTACHED, FILAMENT), "priority"),
    "h_separate": {ATTACHED: "attached", FILAMENT: "filamentous"},
    "h_rare": dict.fromkeys(RARE, "rare"),
    "h_priority_rare": {
        **dict.fromkeys((ATTACHED, FILAMENT), "priority"),
        **dict.fromkeys(RARE, "rare"),
    },
    "h_separate_rare": {
        ATTACHED: "attached",
        FILAMENT: "filamentous",
        **dict.fromkeys(RARE, "rare"),
    },
    "h_common3": dict.fromkeys(("rotifers", "testate_amoebae", "gastrotrichs"), "others"),
    "h_priority_repeat": dict.fromkeys((ATTACHED, FILAMENT), "priority_repeat"),
    "h_rare_repeat": dict.fromkeys(RARE, "rare_repeat"),
    "h_priority_repeat_rare_repeat": {
        **dict.fromkeys((ATTACHED, FILAMENT), "priority_repeat"),
        **dict.fromkeys(RARE, "rare_repeat"),
    },
    "h_separate_rare_repeat": {
        ATTACHED: "attached",
        FILAMENT: "filamentous",
        **dict.fromkeys(RARE, "rare_repeat"),
    },
}
METHODS = ("cosine30", *REPLACEMENTS)


def job_dir(job, reports=False):
    return (REPORTS if reports else OUTPUTS) / f"fold_{job['fold']}" / job["run_id"]


def label_map(names, selected):
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(names):
        raise ValueError("Invalid specialist class subset")
    labels = [i for i, name in enumerate(names, 1) if name in selected]
    return {label: i for i, label in enumerate(labels, 1)}


def routes(names, method):
    if method not in REPLACEMENTS:
        raise ValueError("Unregistered hybrid method")
    return {name: REPLACEMENTS[method].get(name, "general") for name in names}


def combine(sources, route):
    """Exactly one source/class; no cross-class suppression or global score cap."""
    if not set(route.values()) <= set(sources):
        raise ValueError("Missing routed source")
    parts = [
        sources[source].loc[sources[source].class_name == name].copy()
        for name, source in route.items()
    ]
    return pd.concat(parts, ignore_index=True)[components.PREDICTION_COLUMNS]


def routed_thresholds(selections, route):
    return {
        name: selections[source]["policies"]["count_guarded"][name]
        for name, source in route.items()
    }


def subset_annotations(anns, names):
    return anns[(anns.supervision != "target") | anns.class_name.isin(names)].copy()


def assert_membership(frame, ids, names):
    if not set(frame.image_id) <= set(ids) or not set(frame.class_name) <= set(names):
        raise ValueError("Prediction stage/class contamination")


def verify(reg):
    expected = {(s, f) for s in SEEDS for f in range(3)}
    if (
        reg["deadline"] != DEADLINE.isoformat()
        or reg["component_fits"] != 30
        or len(reg["jobs"]) != 6
        or {(j["seed"], j["fold"]) for j in reg["jobs"]} != expected
        or reg["replacements"] != REPLACEMENTS
        or reg["groups"] != {k: list(v) for k, v in GROUPS.items()}
    ):
        raise ValueError("Registered hybrid design/budget changed")
    if reg["package_sha256"] != package_digest():
        raise ValueError("Training package changed")
    for record in reg["inputs"]:
        old.verify_pin(record)
    previous.verify(read_json(previous.REPORTS / "registration.json"))
    for job in reg["jobs"]:
        if set(job["components"]) != set(GROUPS):
            raise ValueError("Component budget changed")
        old.verify_pin(job["config"])
        for record in job["components"].values():
            old.verify_pin(record)
        for source in job["reuse"].values():
            for key in ("config", "model", "calibration_predictions", "outer_predictions"):
                old.verify_pin(source[key])
            for key in ("completion", "calibration"):
                if key in source:
                    old.verify_pin(source[key])


def prepare():
    path = REPORTS / "registration.json"
    if path.exists():
        reg = read_json(path)
        verify(reg)
        return reg
    components.check_deadline(DEADLINE)
    prior = read_json(previous.REPORTS / "registration.json")
    previous.verify(prior)
    REPORTS.mkdir(parents=True, exist_ok=True)
    source = ROOT / "NIGHT_RUN_2026_10_23_HYBRIDS.md"
    protocol = REPORTS / "protocol.md"
    if protocol.exists() and protocol.read_bytes() != source.read_bytes():
        raise ValueError("Refusing protocol overwrite")
    if not protocol.exists():
        protocol.write_bytes(source.read_bytes())
    inputs = [
        old.pin(p)
        for p in (
            source,
            protocol,
            Path(__file__).resolve(),
            ROOT / "tests/test_overnight_hybrids.py",
            previous.REPORTS / "registration.json",
            ROOT / "scripts/overnight_experiments.py",
            ROOT / "scripts/report_hybrid_night.py",
        )
    ]
    jobs = []
    for seed in SEEDS:
        for fold in range(3):
            control = next(j for j in prior["controls"] if j["seed"] == seed and j["fold"] == fold)
            specialist = next(
                j
                for j in prior["jobs"]
                if j["seed"] == seed and j["fold"] == fold and j["recipe"] == "specialists"
            )
            previous.verify_completed(specialist)
            config = load_config(ROOT / control["config"]["path"])
            control_report, control_output = old.run_dirs(config, control["run_id"])
            reuse = {
                "general": {
                    "config": control["config"],
                    "model": old.pin(control_output / "model.pt"),
                    "calibration": old.pin(control_report / "calibration.json"),
                    "calibration_predictions": old.pin(
                        control_output / "predictions_validation.parquet"
                    ),
                    "outer_predictions": old.pin(
                        control_output / "predictions_outer_validation.parquet"
                    ),
                }
            }
            for name in ("filamentous", "others"):
                out = previous.job_dir(specialist)
                reuse[name] = {
                    "config": specialist["components"][name],
                    "model": old.pin(out / name / "model.pt"),
                    "completion": old.pin(out / name / "complete.json"),
                    "calibration_predictions": old.pin(out / f"{name}_calibration.parquet"),
                    "outer_predictions": old.pin(out / f"{name}_outer_validation.parquet"),
                }
            configs = {}
            for name, selected in GROUPS.items():
                raw = config.model_dump(mode="json", exclude={"root"})
                mapping = label_map(config.classes.names, selected)
                raw["classes"]["target"] = [
                    {**t, "label": mapping[t["label"]]}
                    for t in raw["classes"]["target"]
                    if t["label"] in mapping
                ]
                raw["paths"].update(
                    outputs_dir=OUTPUTS.relative_to(ROOT).as_posix(),
                    experiments_dir=REPORTS.relative_to(ROOT).as_posix(),
                )
                if name.endswith("_repeat"):
                    raw["train"]["sampling"] = "repeat_sqrt_020_cap3"
                cfg = REPORTS / "queue_configs" / f"fold_{fold}_{name}_s{seed}.yaml"
                cfg.parent.mkdir(parents=True, exist_ok=True)
                if cfg.exists() and yaml.safe_load(cfg.read_text(encoding="utf-8")) != raw:
                    raise ValueError("Refusing component config overwrite")
                if not cfg.exists():
                    cfg.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
                load_config(cfg)
                configs[name] = old.pin(cfg)
            jobs.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "recipe": "hybrids",
                    "run_id": f"hybrids_s{seed}",
                    "config": control["config"],
                    "components": configs,
                    "reuse": reuse,
                }
            )
    reg = {
        "created_at": night.now().isoformat(),
        "deadline": DEADLINE.isoformat(),
        "component_fits": 30,
        "jobs": jobs,
        "folds": prior["folds"],
        "inputs": inputs,
        "package_sha256": package_digest(),
        "development_images": prior["development_images"],
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "replacements": REPLACEMENTS,
        "primary": "count_guarded",
        "independent_test": False,
        "test_reopened": False,
    }
    verify(reg)
    night.atomic_json(path, reg)
    return reg


def verify_completed(job):
    final = read_json(job_dir(job, True) / "complete.json")
    if final["job"] != job:
        raise ValueError("Completed job identity mismatch")
    for record in final["artifacts"]:
        old.verify_pin(record)
    records = []
    for pin in final["records"]:
        old.verify_pin(pin)
        record = read_json(ROOT / pin["path"])
        old.verify_pin(record["predictions"])
        old.verify_pin(record["calibration"])
        records.append(record)
    if {r["method"] for r in records} != set(REPLACEMENTS):
        raise ValueError("Incomplete hybrid method coverage")
    for name in GROUPS:
        if not components.completed(job_dir(job) / name):
            raise ValueError("Missing completed component")
    return records


def stage_sources(job, configs, samples, stage):
    result = {}
    for name, reuse in job["reuse"].items():
        key = "calibration_predictions" if stage == "calibration" else "outer_predictions"
        old.verify_pin(reuse[key])
        result[name] = pd.read_parquet(ROOT / reuse[key]["path"])
    for name in GROUPS:
        components.check_deadline(DEADLINE)
        result[name] = components.detector_predictions(
            configs[name], job_dir(job) / name, samples, job_dir(job) / f"{name}_{stage}.parquet"
        )
    for name, frame in result.items():
        assert_membership(frame, [s.image_id for s in samples], configs[name].classes.names)
    return result


def worker(index):
    with night.FileLock(night.OUTPUTS / "gpu.lock"):
        components.check_deadline(DEADLINE)
        reg = read_json(REPORTS / "registration.json")
        verify(reg)
        job = reg["jobs"][index]
        output, report = job_dir(job), job_dir(job, True)
        output.mkdir(parents=True, exist_ok=True)
        report.mkdir(parents=True, exist_ok=True)
        config = load_config(ROOT / job["config"]["path"])
        samples, _ = split_samples(config, "train")
        cal_samples, cal_anns = split_samples(config, "validation")
        manifest = load_split(config)
        if not {s.image_id for s in samples} <= set(
            manifest.loc[manifest.split == "train", "image_id"]
        ):
            raise ValueError("FIT contamination")
        configs = {name: load_config(ROOT / p["path"]) for name, p in job["components"].items()}
        configs.update(
            {name: load_config(ROOT / r["config"]["path"]) for name, r in job["reuse"].items()}
        )
        for name, selected in GROUPS.items():
            remapped = components.remap_samples(samples, label_map(config.classes.names, selected))
            weights = sample_weights(remapped, configs[name].train.sampling)
            previous.freeze_json(
                report / f"fit_{name}.json",
                {
                    "fit_ids": [s.image_id for s in remapped],
                    "class_objects": {
                        str(label): sum(s.labels.count(label) for s in remapped)
                        for label in range(1, len(selected) + 1)
                    },
                    "empty_frames": sum(not s.labels for s in remapped),
                    "sampling": configs[name].train.sampling,
                    "weight_min": min(weights) if weights else None,
                    "weight_max": max(weights) if weights else None,
                },
            )
            components.fit_detector(configs[name], remapped, output / name, DEADLINE)

        cal_sources = stage_sources(job, configs, cal_samples, "calibration")
        selections = {}
        for name, cfg in configs.items():
            if name == "general":
                selections[name] = read_json(ROOT / job["reuse"][name]["calibration"]["path"])
            else:
                selections[name] = old.calibrate(
                    cfg,
                    cal_sources[name],
                    subset_annotations(cal_anns, cfg.classes.names),
                    manifest[manifest.split == "validation"],
                )
            selections[name]["source_model"] = (
                job["reuse"][name]["model"]
                if name in job["reuse"]
                else old.pin(output / name / "model.pt")
            )
            selections[name]["source_predictions"] = (
                job["reuse"][name]["calibration_predictions"]
                if name in job["reuse"]
                else old.pin(output / f"{name}_calibration.parquet")
            )
            previous.freeze_json(report / f"source_calibration_{name}.json", selections[name])
        for method in REPLACEMENTS:
            route = routes(config.classes.names, method)
            frame = combine(cal_sources, route)
            selection = old.calibrate(
                config, frame, cal_anns, manifest[manifest.split == "validation"]
            )
            # Preserve unchanged class thresholds even when global fallback would shift.
            selection["diagnostic_joint_class_details"] = selection.pop("class_details")
            selection["policies"]["count_guarded"] = routed_thresholds(selections, route)
            selection["routes"] = route
            selection["primary_source"] = "source_specific_count_guarded"
            selection["source_calibrations"] = {
                name: old.pin(report / f"source_calibration_{name}.json")
                for name in set(route.values())
            }
            previous.freeze_json(report / f"calibration_{method}.json", selection)
        authorization = {
            "calibrations": previous.require_calibration(report, REPLACEMENTS),
            "test_excluded": True,
            "historical_outer_already_inspected": True,
        }
        previous.freeze_json(report / "outer_authorization.json", authorization)
        # New source outer inference and reused outer loading ONLY after all thresholds freeze.
        outer_samples, anns = old.outer_samples(config, reg["folds"][job["fold"]])
        outer_sources = stage_sources(job, configs, outer_samples, "outer_validation")
        records = []
        ids = [s.image_id for s in outer_samples]
        for method in REPLACEMENTS:
            selection = read_json(report / f"calibration_{method}.json")
            frame = combine(outer_sources, selection["routes"])
            out_path = output / f"predictions_{method}_outer_validation.parquet"
            if out_path.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(out_path), frame)
            else:
                components.atomic_parquet(out_path, frame)
            record = {
                "job": job,
                "method": method,
                "predictions": old.pin(out_path),
                "calibration": old.pin(report / f"calibration_{method}.json"),
                "policies": {
                    p: old.measure(
                        config, old.apply_thresholds(frame, selection["policies"][p]), anns, ids
                    )
                    for p in POLICIES
                },
                "diagnostics": components.diagnostics(config, frame, anns),
                "raw_max_detections_per_frame": (
                    int(frame.groupby("image_id").size().max()) if len(frame) else 0
                ),
            }
            previous.freeze_json(report / f"metrics_{method}.json", record)
            records.append(old.pin(report / f"metrics_{method}.json"))
        artifacts = [
            old.pin(p)
            for name in GROUPS
            for p in (
                output / name / "complete.json",
                output / name / "model.pt",
                output / name / "resume_state.pt",
                output / name / "train_log.jsonl",
            )
        ]
        artifacts += [old.pin(p) for p in sorted(output.glob("*.timing.json"))]
        artifacts += [old.pin(p) for p in sorted(report.glob("source_calibration_*.json"))]
        artifacts.append(old.pin(report / "outer_authorization.json"))
        previous.freeze_json(
            report / "complete.json", {"job": job, "records": records, "artifacts": artifacts}
        )


def aggregate(pooled, names):
    means = {}
    for method in METHODS:
        if not all(f"{method}_s{s}" in pooled for s in SEEDS):
            continue
        means[method] = {}
        for policy in POLICIES:
            rows = [pooled[f"{method}_s{s}"][policy] for s in SEEDS]
            classes = []
            for name in names:
                item = {"name": name}
                det = [next(c for c in r["classes"] if c["name"] == name) for r in rows]
                count = [
                    next(c for c in r["counting"]["classes"] if c["name"] == name) for r in rows
                ]
                for key in (
                    "f1",
                    "precision",
                    "recall",
                    "mae_per_frame",
                    "bias_per_frame",
                    "mae_per_frame_all",
                    "bias_per_frame_all",
                ):
                    values = [
                        c[key] for c in (det if key in ("f1", "precision", "recall") else count)
                    ]
                    item[f"mean_{key}"] = statistics.mean(values) if None not in values else None
                    item[f"std_{key}"] = statistics.stdev(values) if None not in values else None
                classes.append(item)
            means[method][policy] = {
                "mean_macro_f1": statistics.mean(r["macro_f1"] for r in rows),
                "std_macro_f1": statistics.stdev(r["macro_f1"] for r in rows),
                "classes": classes,
            }
    return means


def eligibility(means):
    if "cosine30" not in means:
        return {}
    baseline = means["cosine30"]["count_guarded"]
    base = {c["name"]: c for c in baseline["classes"]}
    result = {}
    for method in REPLACEMENTS:
        if method not in means:
            continue
        candidate = means[method]["count_guarded"]
        cs = {c["name"]: c for c in candidate["classes"]}
        criteria = {
            "macro_f1_noninferiority": candidate["mean_macro_f1"]
            >= baseline["mean_macro_f1"] - 0.01,
            "no_class_regression": all(
                cs[n]["mean_f1"] is not None
                and base[n]["mean_f1"] is not None
                and cs[n]["mean_f1"] >= base[n]["mean_f1"] - 0.02
                for n in base
            ),
            **{
                f"count_gain_{n}": cs[n]["mean_mae_per_frame"]
                <= 0.9 * base[n]["mean_mae_per_frame"]
                for n in (ATTACHED, FILAMENT)
            },
        }
        result[method] = {"criteria": criteria, "eligible": all(criteria.values())}
    return result


def baseline_record(job):
    source = job["reuse"]["general"]
    return {
        "job": job,
        "method": "cosine30",
        "predictions": source["outer_predictions"],
        "calibration": source["calibration"],
    }


def summarize(reg, queue):
    records = [baseline_record(j) for j in reg["jobs"]]
    completed = 0
    for job, state in zip(reg["jobs"], queue, strict=True):
        if state["status"] == "completed":
            records.extend(verify_completed(job))
            completed += 1
    pooled, diagnostics = {}, {}
    for seed in SEEDS:
        for method in METHODS:
            selected = [r for r in records if r["job"]["seed"] == seed and r["method"] == method]
            if len(selected) != 3:
                continue
            if {r["job"]["fold"] for r in selected} != {0, 1, 2}:
                raise ValueError("Duplicate/missing OOF folds")
            predictions, annotations, ids, diag = {p: [] for p in POLICIES}, [], [], []
            for record in selected:
                old.verify_pin(record["predictions"])
                old.verify_pin(record["calibration"])
                job = record["job"]
                config = load_config(ROOT / job["config"]["path"])
                samples, anns = old.outer_samples(config, reg["folds"][job["fold"]])
                ids.extend(s.image_id for s in samples)
                annotations.append(anns)
                raw = pd.read_parquet(ROOT / record["predictions"]["path"])
                selection = read_json(ROOT / record["calibration"]["path"])
                diag.append(record.get("diagnostics") or components.diagnostics(config, raw, anns))
                for p in POLICIES:
                    predictions[p].append(old.apply_thresholds(raw, selection["policies"][p]))
            if len(ids) != reg["development_images"] or len(ids) != len(set(ids)):
                raise ValueError("Invalid OOF image coverage")
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
                k: sum(d[k] for d in diag)
                for k in (
                    "localization_tp",
                    "localization_fp",
                    "localization_fn",
                    "duplicate_fp",
                    "ignored",
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
            diagnostics[f"{method}_s{seed}"] = totals
    means = aggregate(pooled, config.classes.names)
    summary = {
        "updated_at": night.now().isoformat(),
        "completed_jobs": completed,
        "budget_jobs": 6,
        "component_fit_budget": 30,
        "candidate_count": len(REPLACEMENTS),
        "fold_results": records,
        "pooled_per_seed": pooled,
        "means": means,
        "eligibility": eligibility(means),
        "localization_diagnostics_per_seed": diagnostics,
        "independent_test": False,
        "test_reopened": False,
        "primary": "count_guarded",
    }
    night.atomic_json(REPORTS / "summary.json", summary)
    return summary


def validate_queue(reg, queue):
    if (
        len(queue) != 6
        or [{k: q[k] for k in j} for j, q in zip(reg["jobs"], queue, strict=True)] != reg["jobs"]
    ):
        raise ValueError("Queue differs from registration")
    if any(q["status"] == "failed" for q in queue):
        raise RuntimeError("Failed job requires inspection; no blind retry")


def read_queue(reg):
    path = OUTPUTS / "queue_state.json"
    return (
        read_json(path)["queue"]
        if path.exists()
        else [{**j, "status": "pending"} for j in reg["jobs"]]
    )


def run():
    components.check_deadline(DEADLINE)
    with night.FileLock(night.OUTPUTS / "queue.lock"):
        with night.FileLock(night.OUTPUTS / "gpu.lock"):
            pass
        reg = prepare()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        queue = read_queue(reg)
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
        path = OUTPUTS / "queue_state.json"
        night.atomic_json(path, state)
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
                    job.update(
                        status=(
                            "completed"
                            if code == 0
                            else ("deadline_stopped" if night.now() >= DEADLINE else "failed")
                        ),
                        exit_code=code,
                    )
                    if code == 0:
                        verify_completed(reg["jobs"][index])
                except subprocess.TimeoutExpired:
                    night.stop_worker(process)
                    job["status"] = "deadline_stopped"
                finally:
                    job["finished_at"] = night.now().isoformat()
                    night.atomic_json(path, state)
            if job["status"] == "failed":
                state.update(
                    status="failed_requires_inspection", finished_at=night.now().isoformat()
                )
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
        print(
            {"jobs": len(reg["jobs"]), "fits": reg["component_fits"], "deadline": reg["deadline"]}
        )
    elif args.run:
        run()
    elif args.summarize:
        with night.FileLock(night.OUTPUTS / "queue.lock"):
            reg = read_json(REPORTS / "registration.json")
            verify(reg)
            summarize(reg, read_queue(reg))
    else:
        worker(args.worker)


if __name__ == "__main__":
    main()
