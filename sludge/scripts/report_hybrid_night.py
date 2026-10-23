"""CPU-only hybrid audit and complete metric appendix, including partial jobs."""

from __future__ import annotations

import argparse
import json
import math

import overnight_hybrids as run
import pandas as pd
import torch

from sludge_micro.config import load_config
from sludge_micro.reporting import read_json
from sludge_micro.runs import split_samples


def number(value):
    return "NA" if value is None else f"{value:.4f}"


def table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
                *["| " + " | ".join(map(str, row)) + " |" for row in rows],
            ]
        )
        + "\n"
    )


def audit(reg, queue, summary):
    torch.set_num_threads(4)
    weights, components, costs = [], [], []
    for job, state in zip(reg["jobs"], queue, strict=True):
        config = load_config(run.ROOT / job["config"]["path"])
        fit, _ = split_samples(config, "train")
        fit_ids = {s.image_id for s in fit}
        output = run.job_dir(job)
        source_params, source_seconds = {"general": 43286903}, {}
        for name, reused in job["reuse"].items():
            if "completion" in reused:
                source_params[name] = read_json(run.ROOT / reused["completion"]["path"])[
                    "parameters"
                ]
            weights.append(
                {
                    "fold": job["fold"],
                    "seed": job["seed"],
                    "source": name,
                    "reused": True,
                    "checkpoint": reused["model"],
                    "config": reused["config"],
                }
            )
        for name in run.GROUPS:
            directory = output / name
            completion = directory / "complete.json"
            resume = directory / "resume_state.pt"
            item = {
                "fold": job["fold"],
                "seed": job["seed"],
                "source": name,
                "completed": completion.exists(),
                "resume_present": resume.exists(),
            }
            if resume.exists():
                checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
                epoch = checkpoint["epoch"]
                if not 1 <= epoch <= 30 or not all(
                    k in checkpoint
                    for k in ("optimizer", "generator", "torch_rng", "numpy_rng", "python_rng")
                ):
                    raise ValueError("Invalid resume state")
                item.update(last_complete_epoch=epoch, resume=run.old.pin(resume))
                if not all(math.isfinite(row["loss"]) for row in checkpoint["history"]):
                    raise ValueError("Nonfinite loss")
            if completion.exists():
                done = read_json(completion)
                if not run.components.completed(directory) or len(done["history"]) != 30:
                    raise ValueError("Invalid completed component")
                if (
                    not resume.exists()
                    or checkpoint["epoch"] != 30
                    or checkpoint["history"] != done["history"]
                ):
                    raise ValueError("Resume/history mismatch")
                if set(done["train_ids"]) != fit_ids or done["batchnorm"]["changed"]:
                    raise ValueError("FIT membership or frozen BN mismatch")
                source_params[name] = done["parameters"]
                seconds = sum(
                    json.loads(line)["seconds"]
                    for line in (directory / "train_log.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if line
                )
                source_seconds[name] = seconds
                item.update(parameters=done["parameters"], train_seconds=seconds)
                weights.append(
                    {
                        "fold": job["fold"],
                        "seed": job["seed"],
                        "source": name,
                        "reused": False,
                        "checkpoint": run.old.pin(directory / "model.pt"),
                        "resume": run.old.pin(resume),
                        "config": job["components"][name],
                    }
                )
            components.append(item)
            if resume.exists():
                del checkpoint
        if state["status"] != "completed":
            continue
        records = run.verify_completed(job)
        general_selection = read_json(run.ROOT / job["reuse"]["general"]["calibration"]["path"])
        general_outer = pd.read_parquet(
            run.ROOT / job["reuse"]["general"]["outer_predictions"]["path"]
        )
        for record in records:
            selection = read_json(run.ROOT / record["calibration"]["path"])
            frame = pd.read_parquet(run.ROOT / record["predictions"]["path"])
            for name, source in selection["routes"].items():
                if source == "general":
                    pd.testing.assert_frame_equal(
                        frame[frame.class_name == name].reset_index(drop=True),
                        general_outer[general_outer.class_name == name].reset_index(drop=True),
                    )
                    if (
                        selection["policies"]["count_guarded"][name]
                        != general_selection["policies"]["count_guarded"][name]
                    ):
                        raise ValueError("Untouched class PRIMARY threshold changed")
            sources = set(selection["routes"].values())
            costs.append(
                {
                    "method": record["method"],
                    "fold": job["fold"],
                    "seed": job["seed"],
                    "sources": sorted(sources),
                    "parameters": sum(source_params[s] for s in sources),
                    "new_training_seconds": sum(source_seconds.get(s, 0) for s in sources),
                    "raw_max_detections_per_frame": record["raw_max_detections_per_frame"],
                    "calibration": record["calibration"],
                }
            )
    result = {
        "checked_at": run.night.now().isoformat(),
        "completed_jobs": summary["completed_jobs"],
        "completed_components": sum(c["completed"] for c in components),
        "components": components,
        "unchanged_classes_verified": True,
        "costs": costs,
        "independent_test": False,
        "test_reopened": False,
        "cost_caveat": (
            "Reused training not charged again; components shared across candidates. "
            "Not comparable latency."
        ),
    }
    run.night.atomic_json(run.REPORTS / "final_integrity.json", result)
    run.night.atomic_json(run.REPORTS / "weight_index.json", weights)
    return result


def render(summary, integrity):
    text = [
        "# Selected-class hybrid metric appendix",
        "",
        f"Completed fold/seed jobs: {summary['completed_jobs']}/6; "
        f"completed new fits: {integrity['completed_components']}/30.",
        "",
        "Adaptive development, NOT independent test. Student_6 primary. "
        "Pooled OOF separately per seed, then mean/sample SD across two seeds. "
        "No duplicated frames pooled across seeds. Count MAE/bias = objects/frame; "
        "positive bias = overcount. _all retains predictions in ignore regions. "
        "NA = undefined. Partial folds are not complete OOF.",
        "",
        "PRIMARY count_guarded preserves source-specific thresholds, "
        "including unchanged general classes. Diagnostic global_f1 recalibrates "
        "the entire hybrid. Each class uses exactly one source, no global top100 cap.",
        "",
    ]
    for policy in run.POLICIES:
        text += [f"## Policy {policy}", ""]
        rows = []
        for method, policies in summary["means"].items():
            mean = policies[policy]
            rows.append(
                [
                    method,
                    *[
                        number(summary["pooled_per_seed"][f"{method}_s{s}"][policy]["macro_f1"])
                        for s in run.SEEDS
                    ],
                    f"{number(mean['mean_macro_f1'])} ± {number(mean['std_macro_f1'])}",
                ]
            )
        text += [table(["Method", "20261022", "20261023", "Macro F1 mean ± SD"], rows), ""]
        keys = (
            "f1",
            "precision",
            "recall",
            "mae_per_frame",
            "bias_per_frame",
            "mae_per_frame_all",
            "bias_per_frame_all",
        )
        for method, policies in summary["means"].items():
            text += [
                f"### {method}: class mean ± SD",
                "",
                table(
                    ["Class", *keys],
                    [
                        [
                            c["name"],
                            *[f"{number(c['mean_' + k])} ± {number(c['std_' + k])}" for k in keys],
                        ]
                        for c in policies[policy]["classes"]
                    ],
                ),
                "",
            ]
        for key, policies in summary["pooled_per_seed"].items():
            metric = policies[policy]
            count = {c["name"]: c for c in metric["counting"]["classes"]}
            text += [
                f"### {key}: pooled OOF",
                "",
                table(
                    ["Class", "Support", *keys],
                    [
                        [
                            c["name"],
                            c["support"],
                            *[number(c[k] if k in c else count[c["name"]][k]) for k in keys],
                        ]
                        for c in metric["classes"]
                    ],
                ),
                "",
            ]
    text += [
        "## Strict eligibility",
        "",
        table(
            ["Candidate", "Eligible", "Criteria"],
            [
                [m, e["eligible"], json.dumps(e["criteria"], ensure_ascii=False)]
                for m, e in summary["eligibility"].items()
            ],
        ),
        "",
        "All criteria required; no post-hoc relaxation for partially replaced priority classes.",
        "",
        "## Raw localization diagnostics (.05, IoU .5)",
        "",
        "Different confidence scales: not matched operating points. "
        "Matched-class accuracy is conditional.",
        "",
    ]
    keys = (
        "localization_precision",
        "localization_recall",
        "duplicate_fp",
        "matched_class_accuracy",
    )
    text += [
        table(
            ["Method/seed", *keys],
            [
                [key, *[number(row[k]) for k in keys]]
                for key, row in summary["localization_diagnostics_per_seed"].items()
            ],
        ),
        "",
        "## Costs by fold/seed (components shared across candidate routes)",
        "",
        table(
            ["Method", "Fold", "Seed", "Parameters", "New train seconds", "Raw max boxes/frame"],
            [
                [
                    c["method"],
                    c["fold"],
                    c["seed"],
                    c["parameters"],
                    number(c["new_training_seconds"]),
                    c["raw_max_detections_per_frame"],
                ]
                for c in integrity["costs"]
            ],
        ),
        "",
        integrity["cost_caveat"],
        "",
        "Exact models/configs/resume hashes: weight_index.json. Per-route thresholds: "
        "fold_N/hybrids_sSEED/calibration_METHOD.json, policies.count_guarded. "
        "All single-fold results (including incomplete OOF sets): summary.json / fold_results.",
        "",
    ]
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-summary", action="store_true")
    args = parser.parse_args()
    with run.night.FileLock(run.night.OUTPUTS / "queue.lock"):
        with run.night.FileLock(run.night.OUTPUTS / "gpu.lock"):
            reg = read_json(run.REPORTS / "registration.json")
            run.verify(reg)
            queue = run.read_queue(reg)
            summary = (
                read_json(run.REPORTS / "summary.json")
                if args.reuse_summary
                else run.summarize(reg, queue)
            )
            integrity = audit(reg, queue, summary)
            (run.REPORTS / "final_metrics.md").write_text(
                render(summary, integrity), encoding="utf-8"
            )
            print(
                {
                    "jobs": summary["completed_jobs"],
                    "fits": integrity["completed_components"],
                    "eligible": summary["eligibility"],
                }
            )


if __name__ == "__main__":
    main()
