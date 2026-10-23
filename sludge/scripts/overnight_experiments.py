"""Bounded, single-GPU development queue. Never reads or evaluates test images.

Run with the project Python. All jobs start from COCO, not microorganism weights.
The separate release baseline is descriptive: its original training IDs are unknown.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    import pandas as pd

    from sludge_micro.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/overnight_2026_10_21.yaml"
REPORTS = ROOT / "experiments_updated_2026_10_02/overnight_2026_10_21"
OUTPUTS = ROOT / "outputs_updated_2026_10_02/overnight_2026_10_21"
DEADLINE = datetime.fromisoformat("2026-10-21T08:00:00+03:00")
SEEDS = (20260929, 20260930, 20260931)
RECIPES = {
    "control15": {"epochs": 15, "augment": {"rotation90": False, "hflip": False, "vflip": False}},
    "rotation15": {"epochs": 15},
    "rotation30": {"epochs": 30},
    "cosine30": {"epochs": 30, "lr_schedule": "cosine", "warmup_epochs": 2},
}
RELEASE_SHA = "0355b1ca23ad3cc88483d06de751f81b64a6a648f6888a2ff4d22fbb95e09af9"


def now() -> datetime:
    return datetime.now().astimezone()


def atomic_json(path: Path, data: Any) -> None:  # noqa: ANN401
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class FileLock:
    """An OS lock, automatically released when a process dies (Windows/POSIX)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a+b")
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)

    def __enter__(self) -> FileLock:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another overnight process holds this lock") from None
        return self

    def __exit__(self, *args: object) -> None:
        self.handle.close()


def validation_counting(
    config: ProjectConfig,
    predictions: pd.DataFrame,
    anns: pd.DataFrame,
    ids: list[str],
    threshold: float,
) -> dict[str, Any]:
    from sludge_micro.evaluate import count_errors, match_all
    from sludge_micro.runs import references

    target, ignore = references(anns)
    det, ref = match_all(predictions, target, ignore, config.evaluation)
    return count_errors(det, ref, ids, config.classes.names, threshold)


def release_baseline() -> None:
    from sludge_micro.archives import sha256_file
    from sludge_micro.config import load_config
    from sludge_micro.infer import Predictor
    from sludge_micro.model import load_trained, torch_device
    from sludge_micro.runs import metrics_record, predict_samples, split_samples
    from sludge_micro.runtime import apply_runtime
    from sludge_micro.split import load_split, lock_path

    config = load_config(BASE)
    apply_runtime(config.runtime)
    weights = ROOT / "weights/microorganisms_fasterrcnn.pt"
    if sha256_file(weights) != RELEASE_SHA:
        raise ValueError("Released checkpoint checksum changed")
    model, meta = load_trained(weights, config)
    for key in ("min_size", "max_size"):
        if meta["config"][key] != getattr(config.model, key):
            raise ValueError("Release resize differs from comparison configuration")
    samples, anns = split_samples(config, "validation")
    predictor = Predictor(model, config.classes.names, 0.0, torch_device(config.runtime.device))
    predictions = predict_samples(predictor, samples)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(OUTPUTS / "release_predictions_validation.parquet", index=False)
    passport = json.loads((ROOT / "model/passport.json").read_text(encoding="utf-8"))
    fixed = metrics_record(
        config, "validation", predictions, anns, passport["detector"]["confidence"]
    )
    fixed["confidence_source"] = "release_passport"
    tuned = metrics_record(config, "validation", predictions, anns, None)
    for record in (fixed, tuned):
        record["images"] = len(samples)
        record["counting"] = validation_counting(
            config, predictions, anns, [s.image_id for s in samples], record["confidence"]
        )
    atomic_json(
        REPORTS / "release_baseline.json",
        {
            "checkpoint_sha256": RELEASE_SHA,
            "split_manifest_sha256": json.loads(lock_path(config).read_text())["manifest_sha256"],
            "validation_ids": sorted(load_split(config).query("split == 'validation'").image_id),
            "independent_of_release_training": None,
            "warning": "Release training IDs unavailable; possible train/validation overlap. "
            "This is NOT independent evidence of superiority over the release.",
            "fixed_release_threshold": fixed,
            "retuned_validation_threshold": tuned,
        },
    )


def worker(run_id: str) -> None:
    import pandas as pd

    from sludge_micro.config import load_config
    from sludge_micro.runs import split_samples
    from sludge_micro.train import run_train

    with FileLock(OUTPUTS / "gpu.lock"):
        if now() >= DEADLINE:
            raise RuntimeError("Overnight deadline reached")
        if run_id == "release_baseline":
            release_baseline()
            return
        config = load_config(REPORTS / "queue_configs" / f"{run_id}.yaml")
        resume = (OUTPUTS / run_id / "resume_state.pt").exists()
        metrics = run_train(config, run_id, resume=resume)
        samples, anns = split_samples(config, "validation")
        predictions = pd.read_parquet(OUTPUTS / run_id / "predictions_validation.parquet")
        atomic_json(
            REPORTS / run_id / "validation_counting.json",
            validation_counting(
                config, predictions, anns, [s.image_id for s in samples], metrics["confidence"]
            ),
        )


def make_queue() -> list[dict[str, Any]]:
    from sludge_micro.config import deep_merge, load_config

    config = load_config(BASE).model_dump(mode="json", exclude={"root"})
    folder = REPORTS / "queue_configs"
    folder.mkdir(parents=True, exist_ok=True)
    queue = [{"run_id": "release_baseline", "status": "pending"}]
    # Complete all four recipes for each seed before moving to the next seed.
    for seed in SEEDS:
        for recipe, training in RECIPES.items():
            run_id = f"{recipe}_s{seed}"
            raw = deep_merge(config, {"train": training, "runtime": {"seed": seed}})
            path = folder / f"{run_id}.yaml"
            if path.exists():
                if yaml.safe_load(path.read_text(encoding="utf-8")) != raw:
                    raise ValueError(f"Refusing to change registered configuration: {path}")
            else:
                path.write_text(
                    yaml.safe_dump(raw, allow_unicode=True, sort_keys=True), encoding="utf-8"
                )
            load_config(path)
            queue.append({"run_id": run_id, "recipe": recipe, "seed": seed, "status": "pending"})
    return queue


def refresh_summary(queue: list[dict[str, Any]]) -> None:
    records = []
    for entry in queue:
        if entry["run_id"] == "release_baseline" or entry["status"] != "completed":
            continue
        metric = json.loads(
            (REPORTS / entry["run_id"] / "metrics.json").read_text(encoding="utf-8")
        )
        records.append(
            {**entry, "macro_f1": metric["macro_f1"], "confidence": metric["confidence"]}
        )
    aggregate = {}
    for recipe in RECIPES:
        values = [r["macro_f1"] for r in records if r["recipe"] == recipe]
        if values:
            aggregate[recipe] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
            }
    atomic_json(
        REPORTS / "summary.json",
        {
            "updated_at": now().isoformat(),
            "deadline": DEADLINE.isoformat(),
            "scope": "Development validation only; thresholds selected on this validation. "
            "Repeated seeds measure training variability, NOT independent dataset uncertainty.",
            "test_reopened": False,
            "runs": records,
            "recipes": aggregate,
        },
    )


def stop_worker(process: subprocess.Popen) -> None:
    """Stop only this worker tree, including the Windows virtualenv redirector child."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    else:
        process.terminate()
    process.wait(timeout=30)


def run_queue() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with FileLock(OUTPUTS / "queue.lock"):
        # A previous supervisor may have exited while its worker is still alive.
        # Refuse restart until that worker exits; do not mark its job as failed.
        with FileLock(OUTPUTS / "gpu.lock"):
            pass
        path = OUTPUTS / "queue_state.json"
        registered = make_queue()
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            queue = state["queue"]
            if [q["run_id"] for q in queue] != [q["run_id"] for q in registered]:
                raise ValueError("Registered queue differs from saved state")
        else:
            queue = registered
        state = {
            "pid": os.getpid(),
            "started_at": now().isoformat(),
            "deadline": DEADLINE.isoformat(),
            "status": "running",
            "queue": queue,
        }
        atomic_json(path, state)
        for entry in queue:
            if now() >= DEADLINE:
                break
            if entry["status"] in {"completed", "failed"}:
                continue
            run_id = entry["run_id"]
            # A parent interruption can happen after the worker writes its final result.
            complete = REPORTS / (
                "release_baseline.json"
                if run_id == "release_baseline"
                else f"{run_id}/validation_counting.json"
            )
            if complete.exists():
                entry["status"] = "completed"
                continue
            entry.update(status="running", started_at=now().isoformat())
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", run_id]
            with (OUTPUTS / f"{run_id}.log").open("a", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUTF8": "1"},
                )
                entry["pid"] = process.pid
                atomic_json(path, state)
                try:
                    result = process.wait(timeout=max(1, (DEADLINE - now()).total_seconds()))
                    entry.update(status="completed" if result == 0 else "failed", exit_code=result)
                except subprocess.TimeoutExpired:
                    # The previous completed epoch checkpoint survives the deadline.
                    stop_worker(process)
                    entry.update(status="deadline_stopped")
                finally:
                    entry["finished_at"] = now().isoformat()
                    atomic_json(path, state)
            refresh_summary(queue)
        state["status"] = "deadline_reached" if now() >= DEADLINE else "queue_finished"
        state["finished_at"] = now().isoformat()
        atomic_json(path, state)
        refresh_summary(queue)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help="Internal worker ID, not a test split selector")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if arguments.worker:
        worker(arguments.worker)
    else:
        run_queue()
