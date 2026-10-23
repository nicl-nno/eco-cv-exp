"""CPU-only final audit and Markdown appendix for the completed second night.

No training, inference, threshold selection or test evaluation. Reads the frozen
registration and existing development results. Does not modify the runner.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any

import overnight_calibrated as study

from sludge_micro.archives import sha256_file
from sludge_micro.config import load_config
from sludge_micro.reporting import read_json
from sludge_micro.runs import run_checks, run_dirs


def audit() -> dict[str, Any]:
    registration = read_json(study.REPORTS / "registration.json")
    study.verify_registration(registration)
    queue = read_json(study.OUTPUTS / "queue_state.json")
    if queue["status"] != "queue_finished" or len(queue["queue"]) != 24:
        raise ValueError("Expected a completed 24-job queue")
    records = []
    for job, registered in zip(queue["queue"], registration["jobs"], strict=True):
        if job["status"] != "completed" or job["exit_code"] != 0:
            raise ValueError("Incomplete or failed job")
        if {k: job[k] for k in registered} != registered:
            raise ValueError("Queue identity differs from registration")
        config = load_config(study.ROOT / job["config"]["path"])
        report, output = run_dirs(config, job["run_id"])
        checks = run_checks(config, job["run_id"])
        if any(v is False for v in checks.values()):
            raise ValueError("Run manifest integrity failure")
        metrics = read_json(report / "outer_metrics.json")
        if metrics["job"] != registered:
            raise ValueError("Outer metrics job identity mismatch")
        for key in ("calibration", "predictions"):
            study.verify_pin(metrics[key])
        history = [
            json.loads(line) for line in (output / "train_log.jsonl").read_text().splitlines()
        ]
        if [r["epoch"] for r in history] != list(range(1, config.train.epochs + 1)):
            raise ValueError("Missing or duplicate completed epochs")
        if not all(math.isfinite(r["loss"]) and math.isfinite(r["lr"]) for r in history):
            raise ValueError("Nonfinite training history")
        resume = output / "resume_state.pt"
        if resume.stat().st_size <= 0:
            raise ValueError("Empty resume checkpoint")
        calibration = study.ROOT / metrics["calibration"]["path"]
        predictions = study.ROOT / metrics["predictions"]["path"]
        if calibration.stat().st_mtime_ns > predictions.stat().st_mtime_ns:
            raise ValueError("Calibration timestamp later than outer predictions")
        records.append(
            {
                "job": registered,
                "checks": checks,
                "epochs": len(history),
                "resume_bytes": resume.stat().st_size,
                "model": (output / "model.pt").relative_to(study.ROOT).as_posix(),
                "calibration": metrics["calibration"],
                "predictions": metrics["predictions"],
            }
        )
    return {
        "checked_at": study.night.now().isoformat(),
        "completed_jobs": len(records),
        "queue_finished_at": queue["finished_at"],
        "registration": study.pin(study.REPORTS / "registration.json"),
        "summary_sha256": sha256_file(study.REPORTS / "summary.json"),
        "resume_contents_deserialized": False,
        "records": records,
    }


def cell(mean: float, sd: float) -> str:
    return f"{mean:.6f} ± {sd:.6f}"


def append_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")


def main() -> None:
    # Shared locks establish that no old or new scientific worker/supervisor is active.
    with (
        study.night.FileLock(study.night.OUTPUTS / "queue.lock"),
        study.night.FileLock(study.night.OUTPUTS / "gpu.lock"),
    ):
        integrity = audit()
        summary = read_json(study.REPORTS / "summary.json")
        if summary["completed_jobs"] != 24 or len(summary["pooled_per_seed"]) != 8:
            raise ValueError("Incomplete aggregation")
        lines = [
            "# Вторая ночь: проверенные численные результаты",
            "",
            f"Проверка: {integrity['checked_at']}. "
            f"Завершение очереди: {integrity['queue_finished_at']}.",
            "",
            "Источник — summary.json; 204 разных development-кадра на каждый pooled OOF seed.",
            "Средние и выборочные SD рассчитаны по двум seed, не по объединённым повторным кадрам.",
            "SD по двум seed не является доверительным интервалом; независимого test здесь нет.",
            "",
            "## Macro F1: все заранее заданные политики",
            "",
        ]
        macro_rows, seed_rows = [], []
        for recipe in study.RECIPES:
            for policy in study.POLICIES:
                result = summary["means"][recipe][policy]
                values = [summary["pooled_per_seed"][f"{recipe}_s{s}"][policy] for s in study.SEEDS]
                if any(v["images"] != 204 for v in values):
                    raise ValueError("Incorrect pooled frame coverage")
                mean = statistics.mean(v["macro_f1"] for v in values)
                sd = statistics.stdev(v["macro_f1"] for v in values)
                if not math.isclose(mean, result["mean_macro_f1"], abs_tol=1e-12):
                    raise ValueError("Mean mismatch")
                if not math.isclose(sd, result["std_macro_f1"], abs_tol=1e-12):
                    raise ValueError("SD mismatch")
                macro_rows.append([recipe, policy, cell(mean, sd)])
            for seed in study.SEEDS:
                pooled = summary["pooled_per_seed"][f"{recipe}_s{seed}"]
                seed_rows.append(
                    [recipe, str(seed), *[f"{pooled[p]['macro_f1']:.6f}" for p in study.POLICIES]]
                )
        append_table(lines, ["Режим", "Политика", "Macro F1, mean ± SD"], macro_rows)
        lines += ["## Pooled OOF отдельно по seed", ""]
        append_table(lines, ["Режим", "Seed", *study.POLICIES], seed_rows)
        for policy in study.POLICIES:
            lines += [
                f"## Поклассовые результаты: {policy}",
                "",
                "MAE и bias — объектов/кадр, включая кадры без объектов. "
                "Bias = prediction − truth.",
                "Обычные MAE/bias исключают предсказания, помеченные evaluator как ignored;",
                "колонки all сохраняют их. Это оценка по разметке, "
                "не независимое измерение истинного количества.",
                "",
            ]
            for recipe in study.RECIPES:
                lines += [f"### {recipe}", ""]
                rows = []
                for c in summary["means"][recipe][policy]["classes"]:
                    rows.append(
                        [
                            c["name"],
                            cell(c["mean_f1"], c["std_f1"]),
                            *[
                                cell(c[f"mean_{k}"], c[f"std_{k}"])
                                for k in (
                                    "mae_per_frame",
                                    "bias_per_frame",
                                    "mae_per_frame_all",
                                    "bias_per_frame_all",
                                )
                            ],
                        ]
                    )
                append_table(lines, ["Класс", "F1", "MAE", "Bias", "MAE all", "Bias all"], rows)
        lines += ["## Проверка заранее заданного критерия", ""]
        for recipe, decision in summary["eligibility"].items():
            lines.append(
                f"- {recipe}: eligible={decision['eligible']}; {json.dumps(decision['criteria'])}."
            )
        lines += [
            "",
            "## Все 24 набора весов",
            "",
            "Пути относительно корня microorganism_detector. Это fold-модели для исследования,",
            "не единая финальная модель для внедрения. Пороги соответствующего запуска лежат в",
            "experiments_updated_2026_10_02/overnight_2026_10_22/fold_N/<run_id>/calibration.json.",
            "",
        ]
        append_table(
            lines,
            ["Весовой файл", "SHA-256", "Эпохи"],
            [
                [f"`{r['model']}`", f"`{r['checks']['checkpoint_sha256']}`", str(r["epochs"])]
                for r in integrity["records"]
            ],
        )
        lines += [
            "Проверены: 24 checkpoint, snapshots, split locks, calibration/prediction hashes;",
            "полные конечные последовательности эпох, конечные loss/LR, непустые resume_state.pt.",
            "Resume-файлы не десериализовались при этой проверке. "
            "Общие queue/GPU OS locks свободны.",
            "",
        ]
        study.night.atomic_json(study.REPORTS / "final_integrity.json", integrity)
        (study.REPORTS / "final_metrics.md").write_text("\n".join(lines), encoding="utf-8")
        print(
            json.dumps(
                {
                    "verified_jobs": 24,
                    "pooled_sets": 8,
                    "eligibility": summary["eligibility"],
                    "finished_at": integrity["queue_finished_at"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
