"""Run a resumable, single-GPU architecture comparison through the night."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "night"
CONFIG = ROOT / "config_night.yaml"
BASELINE = "resnet34_unet"
VARIANTS = ("resnet34_unet_se", "resnet34_unet_aspp", "resnet34_fpn")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def command(stage: str, architecture: str, fold: int, arguments: list[str]) -> None:
    log_dir = RUNS / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{architecture}_fold_{fold}_{stage}.log"
    print(f"{now()} START {stage}: {architecture} fold={fold} log={log}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"{stage} failed ({completed.returncode}); see {log}")
    print(f"{now()} DONE {stage}: {architecture} fold={fold}", flush=True)


def run_one(architecture: str, fold: int, status: dict) -> None:
    fold_dir = RUNS / architecture / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    key = f"{architecture}/fold_{fold}"
    status["active"] = key
    status["updated"] = now()
    write_json(RUNS / "overnight_status.json", status)
    if not (fold_dir / "result.json").is_file():
        command("train", architecture, fold, [
            str(ROOT / "train.py"), "--config", str(CONFIG), "--architecture", architecture,
            "--output-dir", f"runs/night/{architecture}", "--fold", str(fold),
        ])
    if not (fold_dir / "selection.json").is_file():
        command("select", architecture, fold, [
            str(ROOT / "select_checkpoint.py"), "--config", str(CONFIG),
            "--fold-dir", str(fold_dir),
        ])
    tiling = fold_dir / "tiling_comparison.json"
    if not tiling.is_file():
        command("tiling", architecture, fold, [
            str(ROOT / "evaluate_tiling.py"), "--config", str(CONFIG),
            "--checkpoint", str(fold_dir / "selected_full_miou.pt"),
            "--stride", "448", "--tta", "flip4", "--output", str(tiling),
        ])
    selected = json.loads((fold_dir / "selection.json").read_text(encoding="utf-8"))["selected"]
    overlap = json.loads(tiling.read_text(encoding="utf-8"))["overlap"]
    status["runs"][key] = {
        "state": "done", "completed": now(), "selected_epoch": selected["epoch"],
        "active_iou": selected["active_iou"], "mature_iou": selected["mature_iou"],
        "miou": selected["foreground_miou"], "overlap_miou": overlap["foreground_miou"],
    }
    status["active"] = None
    status["updated"] = now()
    write_json(RUNS / "overnight_status.json", status)


def before_deadline(deadline: datetime) -> bool:
    return datetime.now().astimezone() < deadline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--until", default="08:00", help="Local time after which no new run starts")
    args = parser.parse_args()
    hour, minute = map(int, args.until.split(":"))
    current = datetime.now().astimezone()
    deadline = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= current:
        deadline += timedelta(days=1)
    status_path = RUNS / "overnight_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["deadline"] = deadline.isoformat(timespec="seconds")
        status["state"] = "running"
    else:
        status = {
            "started": now(), "deadline": deadline.isoformat(timespec="seconds"),
            "state": "running", "active": None, "runs": {}, "errors": [],
        }
    write_json(status_path, status)
    screening = [(architecture, fold) for fold in (0, 1)
                 for architecture in (BASELINE, *VARIANTS)]
    try:
        for architecture, fold in screening:
            if not before_deadline(deadline):
                break
            try:
                run_one(architecture, fold, status)
            except Exception as exc:
                status["errors"].append({"run": f"{architecture}/fold_{fold}", "error": str(exc), "time": now()})
                status["active"] = None
                write_json(status_path, status)
                print(traceback.format_exc(), flush=True)

        eligible = []
        for architecture in VARIANTS:
            pairs = [(status["runs"].get(f"{architecture}/fold_{fold}"),
                      status["runs"].get(f"{BASELINE}/fold_{fold}")) for fold in (0, 1)]
            if all(variant and baseline for variant, baseline in pairs):
                mean_gain = sum(v["miou"] - b["miou"] for v, b in pairs) / len(pairs)
                eligible.append((mean_gain, architecture))
        if eligible:
            gain, winner = max(eligible)
            status["screening_winner"] = {"architecture": winner, "mean_gain": gain}
            write_json(status_path, status)
            print(f"{now()} SCREENING winner={winner} mean_gain={gain:.6f}", flush=True)
            for fold in (2, 3, 4):
                for architecture in (BASELINE, winner):
                    if not before_deadline(deadline):
                        break
                    try:
                        run_one(architecture, fold, status)
                    except Exception as exc:
                        status["errors"].append({"run": f"{architecture}/fold_{fold}", "error": str(exc), "time": now()})
                        status["active"] = None
                        write_json(status_path, status)
                        print(traceback.format_exc(), flush=True)
                if not before_deadline(deadline):
                    break
    finally:
        status["state"] = "finished"
        status["active"] = None
        status["updated"] = now()
        write_json(status_path, status)
        print(f"{now()} FINISHED completed_runs={len(status['runs'])} errors={len(status['errors'])}", flush=True)


if __name__ == "__main__":
    main()
