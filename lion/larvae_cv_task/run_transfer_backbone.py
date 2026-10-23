"""Resumable external-data and ConvNeXt experiments on two scene folds."""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from train import atomic_json


HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "data_backbone"
STATUS = ROOT / "status.json"
FOLDS = (0, 3)
EXPERIMENTS = (
    ("external_resnet34", "config_external.yaml", "resnet34_unet"),
    ("imagenet_convnext", "config_convnext.yaml", "convnext_tiny_unet"),
)


def save(status: dict) -> None:
    status["updated"] = datetime.now().astimezone().isoformat()
    atomic_json(status, STATUS)


def execute(stage: str, args: list[str], key: str) -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{key.replace('/', '_')}_{stage}.log"
    with path.open("a", encoding="utf-8") as output:
        process = subprocess.run([sys.executable, *args], cwd=HERE,
                                 stdout=output, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{stage} failed ({process.returncode}): {path}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"runs": {}, "errors": []}
    status["state"] = "running"
    try:
        encoder = HERE / "runs" / "external_pretrain" / "encoder.pt"
        if not encoder.exists():
            status["active"] = "external_pretrain"
            save(status)
            execute("train", ["pretrain_external.py"], "external_pretrain")
        for name, config, architecture in EXPERIMENTS:
            for fold in FOLDS:
                key = f"{name}/fold_{fold}"
                directory = ROOT / name / f"fold_{fold}"
                directory.mkdir(parents=True, exist_ok=True)
                status["active"] = key
                save(status)
                try:
                    if not (directory / "result.json").exists():
                        execute("train", ["train.py", "--config", config,
                                          "--output-dir", f"runs/data_backbone/{name}",
                                          "--fold", str(fold)], key)
                    if not (directory / "selection.json").exists():
                        execute("select", ["select_checkpoint.py", "--config", config,
                                           "--fold-dir", str(directory)], key)
                    if not (directory / "tiling_comparison.json").exists():
                        execute("tiling", ["evaluate_tiling.py", "--config", config,
                                           "--checkpoint", str(directory / "selected_full_miou.pt"),
                                           "--stride", "448", "--tta", "flip4", "--output",
                                           str(directory / "tiling_comparison.json")], key)
                    result = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
                    baseline_dir = HERE / "runs" / "scene_cv" / "resnet34_unet" / f"fold_{fold}"
                    baseline = json.loads((baseline_dir / "tiling_comparison.json").read_text(encoding="utf-8"))
                    mine = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                    reference = json.loads((baseline_dir / "result.json").read_text(encoding="utf-8"))
                    if mine["validation_ids"] != reference["validation_ids"]:
                        raise ValueError(f"Validation ID mismatch: {key}")
                    status["runs"][key] = {"state": "done", "epoch": result["epoch"],
                                           "miou": result["overlap"]["foreground_miou"],
                                           "baseline_miou": baseline["overlap"]["foreground_miou"]}
                    print(json.dumps({key: status["runs"][key]}), flush=True)
                except Exception as exc:
                    status["errors"].append({"run": key, "error": str(exc)})
                    traceback.print_exc()
                    raise
                finally:
                    status["active"] = None
                    save(status)
    except Exception as exc:
        if not status["errors"]:
            status["errors"].append({"run": status.get("active"), "error": str(exc)})
        traceback.print_exc()
        raise
    finally:
        status["state"] = "finished"
        status["active"] = None
        save(status)


if __name__ == "__main__":
    main()
