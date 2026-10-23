"""CPU-only final audit/report from frozen third-night artifacts; no inference."""

from __future__ import annotations

import json
import math
import statistics

import numpy as np
import overnight_architectures as run
import pandas as pd
import torch

from sludge_micro.config import load_config
from sludge_micro.model import build_detector
from sludge_micro.reporting import read_json
from sludge_micro.runs import split_samples
from sludge_micro.split import load_split


def stats(values):
    if any(v is None for v in values):
        return {"mean": None, "sd": None}
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values)}


def number(value):
    return "NA" if value is None else f"{value:.4f}"


def mean_sd(mean, sd):
    return f"{number(mean)} ± {number(sd)}"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def log_seconds(path):
    return sum(
        json.loads(line)["seconds"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    )


def collect(reg, queue, summary):
    torch.set_num_threads(4)
    components, paths, training, timings, oracle, agnostic, merges = [], [], {}, {}, {}, {}, {}
    for job in reg["jobs"]:
        run.verify_completed(job)
        output, report = run.job_dir(job), run.job_dir(job, True)
        config = load_config(run.ROOT / job["config"]["path"])
        samples, _ = split_samples(config, "train")
        fit_ids = {s.image_id for s in samples}
        inner = load_split(config)
        cal_ids = set(inner.loc[inner.split == "validation", "image_id"])
        key = f"{job['recipe']}_s{job['seed']}"
        training.setdefault(key, {"seconds": 0.0, "parameters_per_pipeline": 0})
        params = 0
        names = [*job["components"], *(["classifier"] if job["recipe"] == "cascade" else [])]
        for name in names:
            directory = output / name
            completion = read_json(directory / "complete.json")
            expected = 20 if name == "classifier" else 30
            assert len(completion["history"]) == expected
            assert all(math.isfinite(e["loss"]) for e in completion["history"])
            assert completion["batchnorm"]["changed"] == 0
            assert set(completion["train_ids"]) <= fit_ids
            if name != "classifier":
                assert set(completion["train_ids"]) == fit_ids
            resume_path = directory / "resume_state.pt"
            state = torch.load(resume_path, map_location="cpu", weights_only=False)
            assert state["epoch"] == expected
            assert state["history"] == completion["history"]
            assert all(
                k in state
                for k in ("optimizer", "generator", "torch_rng", "numpy_rng", "python_rng")
            )
            del state
            seconds = log_seconds(directory / "train_log.jsonl")
            params += completion["parameters"]
            training[key]["seconds"] += seconds
            item = {
                "fold": job["fold"],
                "seed": job["seed"],
                "method": job["recipe"],
                "component": name,
                "epochs": expected,
                "training_seconds": seconds,
                "parameters": completion["parameters"],
                "checkpoint": run.old.pin(directory / "model.pt"),
                "resume": run.old.pin(resume_path),
                "calibration": run.old.pin(report / f"calibration_{job['recipe']}.json"),
                "finite_losses": True,
                "frozen_bn_unchanged": True,
            }
            components.append(item)
            paths.append(item)
        if training[key]["parameters_per_pipeline"]:
            assert training[key]["parameters_per_pipeline"] == params
        training[key]["parameters_per_pipeline"] = params
        methods = (
            ["cascade"] if job["recipe"] == "cascade" else ["specialists", "specialists_union"]
        )
        authorization = read_json(report / "outer_authorization.json")
        assert authorization["calibrations"] == run.require_calibration(report, methods)
        for method in methods:
            calibration = read_json(report / f"calibration_{method}.json")
            assert set(calibration["calibration_ids"]) == cal_ids
            cp = report / f"calibration_{method}.json"
            pp = output / f"predictions_{method}_outer_validation.parquet"
            assert cp.stat().st_mtime_ns <= pp.stat().st_mtime_ns
        timing_names = (
            ["proposals_outer_validation", "predictions_cascade_outer_validation"]
            if job["recipe"] == "cascade"
            else ["filamentous_outer_validation", "others_outer_validation"]
        )
        timing = [read_json(output / f"{n}.timing.json") for n in timing_names]
        timings.setdefault(key, {"seconds": 0.0, "images": 0})
        timings[key]["seconds"] += sum(t["seconds"] for t in timing)
        timings[key]["images"] += timing[0]["images"]
        if job["recipe"] == "cascade":
            crops = pd.read_parquet(output / "classifier/crops.parquet")
            assert set(crops.image_id) <= fit_ids
            assert set(crops.label) <= set(range(8))
            matrix = np.asarray(timing[1]["oracle_confusion"], dtype=int)
            assert matrix.shape == (8, 8) and matrix.sum() == timing[1]["oracle_gt_crops"]
            oracle.setdefault(str(job["seed"]), np.zeros((8, 8), dtype=int))
            oracle[str(job["seed"])] += matrix
            metric = read_json(report / "metrics_cascade.json")["agnostic_proposals_diagnostics"]
            agnostic.setdefault(str(job["seed"]), {})
            for k in (
                "localization_tp",
                "localization_fp",
                "localization_fn",
                "ignored",
                "duplicate_fp",
            ):
                agnostic[str(job["seed"])][k] = agnostic[str(job["seed"])].get(k, 0) + metric[k]
        else:
            for method, counts in read_json(report / "merge_outer_validation.json").items():
                merge_key = f"{method}_s{job['seed']}"
                merges.setdefault(merge_key, {})
                for k, value in counts.items():
                    merges[merge_key][k] = merges[merge_key].get(k, 0) + value
    assert len(components) == 24
    # Controls are existing checkpoints. Build CPU architecture once for parameter count only.
    control_cfg = load_config(run.ROOT / reg["controls"][0]["config"]["path"])
    control_model = build_detector(control_cfg)
    control_params = sum(p.numel() for p in control_model.parameters())
    del control_model
    for control in reg["controls"]:
        cfg = load_config(run.ROOT / control["config"]["path"])
        report, output = run.old.run_dirs(cfg, control["run_id"])
        key = f"cosine30_s{control['seed']}"
        training.setdefault(key, {"seconds": 0.0, "parameters_per_pipeline": control_params})
        training[key]["seconds"] += log_seconds(output / "train_log.jsonl")
        paths.append(
            {
                "method": "cosine30",
                "fold": control["fold"],
                "seed": control["seed"],
                "checkpoint": run.old.pin(output / "model.pt"),
                "calibration": run.old.pin(report / "calibration.json"),
                "reused_control": True,
            }
        )
    oracle_results = {}
    for seed, matrix in oracle.items():
        assert matrix.sum() == 707
        oracle_results[seed] = {
            "objects": int(matrix.sum()),
            "correct": int(np.trace(matrix)),
            "accuracy": float(np.trace(matrix) / matrix.sum()),
            "background_rejections": int(matrix[:, 0].sum()),
            "background_rejection_rate": float(matrix[:, 0].sum() / matrix.sum()),
            "confusion": matrix.tolist(),
        }
    for row in agnostic.values():
        tp, fp, fn = (row[k] for k in ("localization_tp", "localization_fp", "localization_fn"))
        row.update(localization_precision=tp / (tp + fp), localization_recall=tp / (tp + fn))
    diagnostic_means = {}
    for method in run.METHODS:
        rows = [summary["localization_diagnostics_per_seed"][f"{method}_s{s}"] for s in run.SEEDS]
        diagnostic_means[method] = {k: stats([row[k] for row in rows]) for k in rows[0]}
    cost_means = {}
    for method in ("cosine30", "cascade", "specialists"):
        rows = [training[f"{method}_s{s}"] for s in run.SEEDS]
        cost_means[method] = {
            "train_seconds_per_three_folds": stats([row["seconds"] for row in rows]),
            "parameters_per_pipeline": rows[0]["parameters_per_pipeline"],
        }
        if method != "cosine30":
            cost_means[method]["outer_seconds_per_image"] = stats(
                [
                    timings[f"{method}_s{s}"]["seconds"] / timings[f"{method}_s{s}"]["images"]
                    for s in run.SEEDS
                ]
            )
    diagnostics = {
        "localization_per_seed": summary["localization_diagnostics_per_seed"],
        "localization_mean_sd": diagnostic_means,
        "agnostic_proposals_per_seed": agnostic,
        "oracle_class_order": ["background", *control_cfg.classes.names],
        "oracle_per_seed": oracle_results,
        "oracle_accuracy_mean_sd": stats([o["accuracy"] for o in oracle_results.values()]),
        "oracle_background_rate_mean_sd": stats(
            [o["background_rejection_rate"] for o in oracle_results.values()]
        ),
        "specialist_merges_per_seed": merges,
        "training_per_seed": training,
        "outer_timing_per_seed": timings,
        "cost_mean_sd": cost_means,
        "timing_caveat": (
            "Model loading included; cascade outer includes oracle GT crops; "
            "specialist merge time excluded. No matched control inference timing. "
            "Not a production latency benchmark."
        ),
    }
    integrity = {
        "checked_at": run.night.now().isoformat(),
        "registration": run.old.pin(run.REPORTS / "registration.json"),
        "summary": run.old.pin(run.REPORTS / "summary.json"),
        "completed_jobs": 12,
        "component_fits": 24,
        "controls_reused": 6,
        "components": components,
        "queue_finished_at": queue["finished_at"],
        "shared_locks_free": True,
        "frozen_inputs_verified": True,
        "calibration_ids_verified": True,
        "calibration_before_outer_mtime_verified": True,
        "fit_crop_ids_verified": True,
        "all_epoch_checkpoints_readable": True,
        "test_reopened": False,
        "independent_test": False,
        "variant": "prefer_student_6",
    }
    return integrity, diagnostics, paths


def metrics_document(summary, diagnostics, paths):
    text = [
        "# Third-night complete metric appendix",
        "",
        "CPU-only reporting of stored predictions/metrics. No new inference or training.",
        "",
        "Each seed has pooled OOF predictions for204 distinct development images; "
        "seeds are NOT concatenated. Mean ± sample SD uses exactly two seed-level values, "
        "not fold averages. NA means undefined. MAE/bias are objects per frame; "
        "positive bias is overcount. '_all' retains detections in ignore regions. "
        "These previously inspected outer folds are adaptive development, not independent test.",
        "",
    ]
    for policy in run.POLICIES:
        text += [
            f"## Policy: {policy}",
            "",
            "PRIMARY" if policy == "count_guarded" else "Diagnostic only",
            "",
        ]
        rows = []
        for method in run.METHODS:
            values = [
                summary["pooled_per_seed"][f"{method}_s{s}"][policy]["macro_f1"] for s in run.SEEDS
            ]
            mean = summary["means"][method][policy]
            rows.append(
                [method, *map(number, values), mean_sd(mean["mean_macro_f1"], mean["std_macro_f1"])]
            )
        text += [table(["Method", *map(str, run.SEEDS), "Macro F1 mean ± SD"], rows), ""]
        for method in run.METHODS:
            text += [f"### {method}: class mean ± SD", ""]
            rows = []
            for c in summary["means"][method][policy]["classes"]:
                rows.append(
                    [
                        c["name"],
                        *[
                            mean_sd(c[f"mean_{k}"], c[f"std_{k}"])
                            for k in (
                                "f1",
                                "precision",
                                "recall",
                                "mae_per_frame",
                                "bias_per_frame",
                                "mae_per_frame_all",
                                "bias_per_frame_all",
                            )
                        ],
                    ]
                )
            text += [
                table(
                    ["Class", "F1", "Precision", "Recall", "MAE", "Bias", "MAE_all", "Bias_all"],
                    rows,
                ),
                "",
            ]
            for seed in run.SEEDS:
                metric = summary["pooled_per_seed"][f"{method}_s{seed}"][policy]
                counts = {c["name"]: c for c in metric["counting"]["classes"]}
                rows = [
                    [
                        c["name"],
                        c["support"],
                        *[number(c[k]) for k in ("f1", "precision", "recall")],
                        *[
                            number(counts[c["name"]][k])
                            for k in (
                                "mae_per_frame",
                                "bias_per_frame",
                                "mae_per_frame_all",
                                "bias_per_frame_all",
                            )
                        ],
                    ]
                    for c in metric["classes"]
                ]
                text += [
                    f"#### Seed {seed}, pooled OOF",
                    "",
                    table(
                        [
                            "Class",
                            "Support",
                            "F1",
                            "Precision",
                            "Recall",
                            "MAE",
                            "Bias",
                            "MAE_all",
                            "Bias_all",
                        ],
                        rows,
                    ),
                    "",
                ]
    text += [
        "## Localization diagnostics at fixed score0.05, IoU0.5",
        "",
        "These scores are not the calibrated PRIMARY scores; "
        "score scales differ across architectures. "
        "Matched-class accuracy is conditional on surviving boxes and is NOT oracle crop accuracy.",
        "",
    ]
    keys = (
        "localization_tp",
        "localization_fp",
        "localization_fn",
        "duplicate_fp",
        "ignored",
        "localization_precision",
        "localization_recall",
        "matched_class_accuracy",
    )
    rows = [
        [key, *[number(row[k]) for k in keys]]
        for key, row in diagnostics["localization_per_seed"].items()
    ]
    text += [table(["Method/seed", *keys], rows), ""]
    rows = [
        [method, *[mean_sd(row[k]["mean"], row[k]["sd"]) for k in keys]]
        for method, row in diagnostics["localization_mean_sd"].items()
    ]
    text += [table(["Mean ± SD", *keys], rows), "", "### Agnostic proposals before classifier", ""]
    rows = [
        [seed, *[number(row[k]) for k in keys[:-1]]]
        for seed, row in diagnostics["agnostic_proposals_per_seed"].items()
    ]
    text += [
        table(["Seed", *keys[:-1]], rows),
        "",
        "## Oracle GT-crop classifier (diagnostic only)",
        "",
        "These outer GT-crop labels never train or calibrate the classifier. "
        "Matrix rows=true classes, columns=predicted classes; background truth row is zero.",
        "",
    ]
    for seed, row in diagnostics["oracle_per_seed"].items():
        text += [
            f"### Seed {seed}: accuracy {number(row['accuracy'])}, "
            f"background rejection {number(row['background_rejection_rate'])}",
            "",
            table(
                ["Truth / predicted", *diagnostics["oracle_class_order"]],
                [
                    [name, *values]
                    for name, values in zip(
                        diagnostics["oracle_class_order"], row["confusion"], strict=True
                    )
                ],
            ),
            "",
        ]
    text += ["## Costs", "", diagnostics["timing_caveat"], ""]
    rows = []
    for method, row in diagnostics["cost_mean_sd"].items():
        train = row["train_seconds_per_three_folds"]
        infer = row.get("outer_seconds_per_image")
        rows.append(
            [
                method,
                row["parameters_per_pipeline"],
                mean_sd(train["mean"], train["sd"]),
                mean_sd(infer["mean"], infer["sd"]) if infer else "not measured comparably",
            ]
        )
    text += [
        table(
            [
                "Method",
                "Parameters/pipeline",
                "Train seconds/3 folds, mean ± SD",
                "Outer seconds/frame",
            ],
            rows,
        ),
        "",
        "## Weights and calibration",
        "",
        "Paths relative to microorganism_detector. See weight_index.json for SHA256. "
        "New pipeline calibration files are in the report folder for the same fold/run: "
        "calibration_cascade.json, calibration_specialists.json, "
        "calibration_specialists_union.json.",
        "",
    ]
    text += [
        table(
            ["Method", "Fold", "Seed", "Component", "Checkpoint"],
            [
                [
                    p["method"],
                    p["fold"],
                    p["seed"],
                    p.get("component", "shared"),
                    p["checkpoint"]["path"],
                ]
                for p in paths
            ],
        )
    ]
    return "\n".join(text)


def main():
    with run.night.FileLock(run.night.OUTPUTS / "queue.lock"):
        with run.night.FileLock(run.night.OUTPUTS / "gpu.lock"):
            reg = read_json(run.REPORTS / "registration.json")
            run.verify(reg)
            queue = read_json(run.OUTPUTS / "queue_state.json")
            run.validate_queue(reg, queue["queue"])
            assert queue["status"] == "queue_finished"
            assert all(j["status"] == "completed" for j in queue["queue"])
            summary = read_json(run.REPORTS / "summary.json")
            assert summary["completed_pipeline_jobs"] == 12
            assert len(summary["pooled_per_seed"]) == 8
            assert run.eligibility(summary["means"]) == summary["eligibility"]
            for row in summary["pooled_per_seed"].values():
                assert all(p["images"] == 204 and p["objects"] == 707 for p in row.values())
            integrity, diagnostics, paths = collect(reg, queue, summary)
            run.night.atomic_json(run.REPORTS / "final_integrity.json", integrity)
            run.night.atomic_json(run.REPORTS / "final_diagnostics.json", diagnostics)
            run.night.atomic_json(run.REPORTS / "weight_index.json", {"weights": paths})
            (run.REPORTS / "final_metrics.md").write_text(
                metrics_document(summary, diagnostics, paths), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "audit": "passed",
                        "components": 24,
                        "eligibility": summary["eligibility"],
                        "oracle_accuracy": diagnostics["oracle_accuracy_mean_sd"],
                        "oracle_background": diagnostics["oracle_background_rate_mean_sd"],
                        "agnostic_proposals": diagnostics["agnostic_proposals_per_seed"],
                        "cost": diagnostics["cost_mean_sd"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
