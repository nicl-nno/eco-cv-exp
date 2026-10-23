"""Sequential, resumable H10/H13 scene-CV experiments after the requested start time."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import yaml

from train import atomic_json


HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "night_hypotheses"
STATUS = ROOT / "status.json"
BASE = HERE / "config_convnext.yaml"
EXPERIMENTS = (
    ("h10_encoder_lr", HERE / "config_h10_encoder_lr.yaml", "encoder_learning_rate"),
    ("h13_scale", HERE / "config_h13_scale.yaml", "augmentation_scale_range"),
)
PILOT = (0, 3)
REMAINDER = (1, 2, 4)


def preflight() -> None:
    baseline = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    if baseline.get("architecture") != "convnext_tiny_unet":
        raise ValueError("Unexpected baseline architecture")
    for name, config_file, changed in EXPERIMENTS:
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        difference = {key for key in baseline.keys() | config.keys()
                      if baseline.get(key) != config.get(key)}
        if difference != {"output_dir", changed}:
            raise ValueError(f"{name}: unexpected config differences: {sorted(difference)}")
        if Path(config["output_dir"]).parts[:2] != ("runs", "night_hypotheses"):
            raise ValueError(f"{name}: unexpected output directory")
    for fold in range(5):
        directory = HERE / "runs" / "data_backbone" / "imagenet_convnext" / f"fold_{fold}"
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        tiling = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
        if not result["train_ids"] or not result["validation_ids"]:
            raise ValueError(f"Empty baseline split on fold {fold}")
        if tiling["overlap_tta"] != "flip4" or tiling["overlap_stride"] != 448:
            raise ValueError(f"Unexpected baseline evaluation on fold {fold}")


def save(status: dict) -> None:
    status["updated"] = datetime.now().astimezone().isoformat()
    atomic_json(status, STATUS)


def run_stage(stage: str, arguments: list[str], name: str, fold: int) -> None:
    log = ROOT / "logs" / f"{name}_fold_{fold}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        result = subprocess.run([sys.executable, *arguments], cwd=HERE,
                                stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{stage} failed ({result.returncode}): {log}")


def evaluate(name: str, config: Path, fold: int, status: dict) -> None:
    key = f"{name}/fold_{fold}"
    directory = ROOT / name / f"fold_{fold}"
    baseline_dir = HERE / "runs" / "data_backbone" / "imagenet_convnext" / f"fold_{fold}"
    status["active"] = key
    save(status)
    if not (directory / "result.json").exists():
        run_stage("train", ["train.py", "--config", str(config), "--fold", str(fold)], name, fold)
    if not (directory / "selection.json").exists():
        run_stage("select", ["select_checkpoint.py", "--config", str(config),
                             "--fold-dir", str(directory)], name, fold)
    if not (directory / "tiling_comparison.json").exists():
        run_stage("tiling", ["evaluate_tiling.py", "--config", str(config),
                             "--checkpoint", str(directory / "selected_full_miou.pt"),
                             "--stride", "448", "--tta", "flip4", "--output",
                             str(directory / "tiling_comparison.json")], name, fold)
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    baseline = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
    metric = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
    reference = json.loads((baseline_dir / "tiling_comparison.json").read_text(encoding="utf-8"))
    if result["train_ids"] != baseline["train_ids"] or result["validation_ids"] != baseline["validation_ids"]:
        raise ValueError(f"Train/validation IDs differ: {key}")
    if metric["overlap_stride"] != 448 or metric["overlap_tta"] != "flip4":
        raise ValueError(f"Evaluation protocol differs: {key}")
    status["runs"][key] = {
        "state": "done", "epoch": metric["epoch"],
        "foreground_miou": metric["overlap"]["foreground_miou"],
        "baseline_miou": reference["overlap"]["foreground_miou"],
        "delta": metric["overlap"]["foreground_miou"] - reference["overlap"]["foreground_miou"],
        "seconds": metric["overlap_seconds"], "baseline_seconds": reference["overlap_seconds"],
    }
    print(json.dumps({key: status["runs"][key]}), flush=True)
    status["active"] = None
    save(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    preflight()
    if args.preflight:
        print("H10/H13 configs and five baseline folds verified")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"runs": {}, "errors": []}
    status["state"] = "running"
    save(status)
    try:
        for name, config, _ in EXPERIMENTS:
            for fold in PILOT:
                evaluate(name, config, fold, status)
            changes = [status["runs"][f"{name}/fold_{fold}"]["delta"] for fold in PILOT]
            promising = sum(changes) / len(changes) >= 0.01 and min(changes) >= -0.01
            status.setdefault("decisions", {})[name] = {
                "pilot_deltas": changes, "extend_to_five_folds": promising,
            }
            save(status)
            if promising:
                for fold in REMAINDER:
                    evaluate(name, config, fold, status)
        status["state"] = "finished"
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
