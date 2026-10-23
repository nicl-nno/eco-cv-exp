"""Resume a bounded, scene-grouped comparison in independent output dirs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from train import atomic_json


HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "scene_cv"
CONFIG = HERE / "config_scene.yaml"
ARCHITECTURES = ("resnet34_unet", "resnet34_unet_se")


def run(stage: str, args: list[str], architecture: str, fold: int) -> None:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{architecture}_fold_{fold}_{stage}.log"
    with path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, *args], cwd=HERE,
                                   stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with {completed.returncode}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 3])
    args = parser.parse_args()
    if any(not 0 <= fold < 5 for fold in args.folds):
        parser.error("folds must be 0–4")
    status_path = ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"runs": {}, "errors": []}
    status["state"] = "running"
    try:
        for fold in args.folds:
            for architecture in ARCHITECTURES:
                key = f"{architecture}/fold_{fold}"
                directory = ROOT / architecture / f"fold_{fold}"
                directory.mkdir(parents=True, exist_ok=True)
                status["active"] = key
                status["updated"] = datetime.now().astimezone().isoformat()
                atomic_json(status, status_path)
                try:
                    if not (directory / "result.json").exists():
                        run("train", ["train.py", "--config", str(CONFIG), "--architecture", architecture,
                                      "--output-dir", f"runs/scene_cv/{architecture}", "--fold", str(fold)],
                            architecture, fold)
                    if not (directory / "selection.json").exists():
                        run("select", ["select_checkpoint.py", "--config", str(CONFIG),
                                       "--fold-dir", str(directory)], architecture, fold)
                    if not (directory / "tiling_comparison.json").exists():
                        run("tiling", ["evaluate_tiling.py", "--config", str(CONFIG),
                                       "--checkpoint", str(directory / "selected_full_miou.pt"),
                                       "--stride", "448", "--tta", "flip4", "--output",
                                       str(directory / "tiling_comparison.json")], architecture, fold)
                    result = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
                    status["runs"][key] = {"state": "done", "epoch": result["epoch"],
                                           "miou": result["overlap"]["foreground_miou"]}
                    print(json.dumps({key: status["runs"][key]}), flush=True)
                except Exception as exc:
                    status["errors"].append({"run": key, "error": str(exc)})
                    traceback.print_exc()
                    raise
                finally:
                    status["active"] = None
                    status["updated"] = datetime.now().astimezone().isoformat()
                    atomic_json(status, status_path)
    finally:
        status["state"] = "finished"
        status["active"] = None
        status["updated"] = datetime.now().astimezone().isoformat()
        atomic_json(status, status_path)


if __name__ == "__main__":
    main()
