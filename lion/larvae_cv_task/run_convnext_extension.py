"""Complete the paired scene-fold comparison without repeating folds 0 and 3."""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch

from train import atomic_json


HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "convnext_extension"
STATUS = ROOT / "status.json"
FOLDS = (1, 2, 4)
MODELS = (
    ("resnet34_unet", "resnet34_unet", "config_scene.yaml", "runs/scene_cv/resnet34_unet"),
    ("imagenet_convnext", "convnext_tiny_unet", "config_convnext.yaml", "runs/data_backbone/imagenet_convnext"),
)


def verify_architecture(path: Path, expected: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actual = checkpoint.get("config", {}).get("architecture")
    if actual != expected:
        raise ValueError(f"Architecture mismatch in {path}: expected {expected}, found {actual}")


def save(status: dict) -> None:
    status["updated"] = datetime.now().astimezone().isoformat()
    atomic_json(status, STATUS)


def execute(stage: str, args: list[str], key: str) -> None:
    log = ROOT / "logs" / f"{key.replace('/', '_')}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.run([sys.executable, *args], cwd=HERE,
                                 stdout=output, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{stage} failed ({process.returncode}): {log}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    status = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {"runs": {}, "errors": []}
    status["state"] = "running"
    save(status)
    try:
        for fold in FOLDS:
            for name, architecture, config, relative_dir in MODELS:
                key = f"{name}/fold_{fold}"
                directory = HERE / relative_dir / f"fold_{fold}"
                directory.mkdir(parents=True, exist_ok=True)
                status["active"] = key
                save(status)
                try:
                    for checkpoint_name in ("last.pt", "best.pt", "selected_full_miou.pt"):
                        checkpoint_path = directory / checkpoint_name
                        if checkpoint_path.exists():
                            verify_architecture(checkpoint_path, architecture)
                    if not (directory / "result.json").exists():
                        execute("train", ["train.py", "--config", config,
                                          "--architecture", architecture,
                                          "--output-dir", relative_dir, "--fold", str(fold)], key)
                    verify_architecture(directory / "best.pt", architecture)
                    if not (directory / "selection.json").exists():
                        execute("select", ["select_checkpoint.py", "--config", config,
                                           "--fold-dir", str(directory)], key)
                    verify_architecture(directory / "selected_full_miou.pt", architecture)
                    if not (directory / "tiling_comparison.json").exists():
                        execute("tiling", ["evaluate_tiling.py", "--config", config,
                                           "--checkpoint", str(directory / "selected_full_miou.pt"),
                                           "--stride", "448", "--tta", "flip4", "--output",
                                           str(directory / "tiling_comparison.json")], key)
                    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                    tiling = json.loads((directory / "tiling_comparison.json").read_text(encoding="utf-8"))
                    if name == "imagenet_convnext":
                        baseline = json.loads((HERE / "runs" / "scene_cv" / "resnet34_unet" /
                                               f"fold_{fold}" / "result.json").read_text(encoding="utf-8"))
                        if result["validation_ids"] != baseline["validation_ids"] or result["train_ids"] != baseline["train_ids"]:
                            raise ValueError(f"Train or validation IDs differ: {key}")
                    if tiling["overlap_stride"] != 448 or tiling["overlap_tta"] != "flip4":
                        raise ValueError(f"Unexpected evaluation protocol: {key}")
                    status["runs"][key] = {"state": "done", "epoch": tiling["epoch"],
                                            "miou": tiling["overlap"]["foreground_miou"]}
                    print(json.dumps({key: status["runs"][key]}), flush=True)
                except Exception as exc:
                    status["errors"].append({"run": key, "error": str(exc)})
                    traceback.print_exc()
                    raise
                finally:
                    status["active"] = None
                    save(status)
        status["state"] = "finished"
    except Exception:
        status["state"] = "failed"
        raise
    finally:
        status["active"] = None
        save(status)


if __name__ == "__main__":
    main()
