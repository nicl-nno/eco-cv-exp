"""Sequential scene-CV pilots; no new training starts after 09:00 Moscow time."""
from __future__ import annotations

import argparse
import json
import msvcrt
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from train import atomic_json

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "morning_hypotheses"
STATUS = ROOT / "status.json"
BASE = HERE / "config_convnext.yaml"
EXPERIMENTS = (
    ("h17_convnext_fpn", "config_h17_convnext_fpn.yaml", "architecture"),
    ("h18_convnext_se", "config_h18_convnext_se.yaml", "architecture"),
    ("h19_color", "config_h19_color.yaml", "color_jitter_probability"),
    ("h20_no_elastic", "config_h20_no_elastic.yaml", "elastic_probability"),
)
PILOT = (0, 3)
REMAINDER = (1, 2, 4)
MOSCOW = timezone(timedelta(hours=3), "MSK")
CUTOFF = datetime(2026, 10, 20, 9, 0, tzinfo=MOSCOW)


def preflight() -> None:
    baseline = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    for name, filename, changed in EXPERIMENTS:
        config = yaml.safe_load((HERE / filename).read_text(encoding="utf-8"))
        difference = {key for key in baseline.keys() | config.keys()
                      if baseline.get(key) != config.get(key)}
        if difference != {"output_dir", changed}:
            raise ValueError(f"{name}: unexpected config differences: {sorted(difference)}")
        if config["output_dir"] != f"runs/morning_hypotheses/{name}":
            raise ValueError(f"{name}: output directory differs")
    for fold in range(5):
        directory = HERE / "runs" / "data_backbone" / "imagenet_convnext" / f"fold_{fold}"
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        tiling = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
        if not result["train_ids"] or not result["validation_ids"]:
            raise ValueError(f"Empty baseline split: {fold}")
        if (tiling["overlap_tta"], tiling["overlap_stride"]) != ("flip4", 448):
            raise ValueError(f"Unexpected baseline protocol: {fold}")


def save(status: dict) -> None:
    status["updated"] = datetime.now(MOSCOW).isoformat()
    atomic_json(status, STATUS)


def run_stage(name: str, fold: int, stage: str, arguments: list[str]) -> None:
    log = ROOT / "logs" / f"{name}_fold_{fold}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        finished = subprocess.run([sys.executable, *arguments], cwd=HERE,
                                  stdout=output, stderr=subprocess.STDOUT)
    if finished.returncode:
        raise RuntimeError(f"{stage} failed ({finished.returncode}): {log}")


def evaluate(name: str, filename: str, fold: int, status: dict) -> bool:
    key = f"{name}/fold_{fold}"
    directory = ROOT / name / f"fold_{fold}"
    baseline_dir = HERE / "runs" / "data_backbone" / "imagenet_convnext" / f"fold_{fold}"
    config = HERE / filename
    if key in status["runs"]:
        return True
    if not (directory / "result.json").is_file() and datetime.now(MOSCOW) >= CUTOFF:
        status.setdefault("deferred", []).append(key)
        save(status)
        return False
    status["active"] = key
    save(status)
    if not (directory / "result.json").is_file():
        run_stage(name, fold, "train", ["train.py", "--config", str(config), "--fold", str(fold)])
    if not (directory / "selection.json").is_file():
        run_stage(name, fold, "select", ["select_checkpoint.py", "--config", str(config),
                                         "--fold-dir", str(directory)])
    if not (directory / "tiling_comparison.json").is_file():
        run_stage(name, fold, "tiling", ["evaluate_tiling.py", "--config", str(config),
                                         "--checkpoint", str(directory / "selected_full_miou.pt"),
                                         "--stride", "448", "--tta", "flip4", "--output",
                                         str(directory / "tiling_comparison.json")])
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    baseline = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
    metric = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
    reference = json.loads((baseline_dir / "tiling_comparison.json").read_text(encoding="utf-8"))
    if (result["train_ids"] != baseline["train_ids"]
            or result["validation_ids"] != baseline["validation_ids"]):
        raise ValueError(f"Train/validation IDs differ: {key}")
    if (metric["overlap_tta"], metric["overlap_stride"]) != ("flip4", 448):
        raise ValueError(f"Evaluation protocol differs: {key}")
    status["runs"][key] = {
        "state": "done", "epoch": metric["epoch"],
        "foreground_miou": metric["overlap"]["foreground_miou"],
        "baseline_miou": reference["overlap"]["foreground_miou"],
        "delta": metric["overlap"]["foreground_miou"] - reference["overlap"]["foreground_miou"],
        "seconds": metric["overlap_seconds"],
        "baseline_seconds": reference["overlap_seconds"],
    }
    print(json.dumps({key: status["runs"][key]}), flush=True)
    status["active"] = None
    save(status)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight()
    if args.preflight:
        print("Four single-factor configs, five baseline folds and Flip4/448 verified")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "runner.lock").open("a+b") as lock:
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Morning runner is already active") from exc
        status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {
            "runs": {}, "errors": [], "decisions": {}, "deferred": [],
        }
        if status.get("state") == "finished":
            print("Morning runner already finished", flush=True)
            return
        if status.get("state") == "failed" and status.get("errors"):
            status.setdefault("recovered_errors", []).extend(status["errors"])
            status["errors"] = []
        status["state"] = "running"
        save(status)
        try:
            for name, filename, _ in EXPERIMENTS:
                for fold in PILOT:
                    evaluate(name, filename, fold, status)
                if all(f"{name}/fold_{fold}" in status["runs"] for fold in PILOT):
                    changes = [status["runs"][f"{name}/fold_{fold}"]["delta"] for fold in PILOT]
                    promising = sum(changes) / 2 >= 0.01 and min(changes) >= -0.01
                    status["decisions"][name] = {
                        "pilot_deltas": changes, "extend_to_five_folds": promising,
                    }
                    save(status)
            for name, filename, _ in EXPERIMENTS:
                if status["decisions"].get(name, {}).get("extend_to_five_folds"):
                    for fold in REMAINDER:
                        evaluate(name, filename, fold, status)
            status["state"] = "time_cutoff" if status["deferred"] else "finished"
        except Exception as exc:
            status["state"] = "failed"
            status["errors"].append({"active": status.get("active"), "error": str(exc)})
            traceback.print_exc()
            raise
        finally:
            status["active"] = None
            save(status)


if __name__ == "__main__":
    main()
