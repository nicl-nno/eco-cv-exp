"""Rank saved epochs by the actual full-image foreground mIoU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from data import read_coco
from model import build_model
from train import atomic_json, atomic_torch_save, full_image_metrics, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    args = parser.parse_args()
    config_file = args.config.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = resolve_path(config_file, config["data_root"])
    images, annotations = read_coco(root)
    paths = [args.fold_dir / "best.pt", args.fold_dir / "last.pt", *sorted(args.fold_dir.glob("candidate_epoch_*.pt"))]
    if len(paths) == 2:
        parser.error("No candidate epochs found; enable candidate_every and train the fold")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_checkpoint = torch.load(paths[0], map_location="cpu", weights_only=False)
    architecture = first_checkpoint.get("config", {}).get("architecture", "resnet34_unet")
    model = build_model(architecture, pretrained=False).to(device).eval()
    rows = []
    expected_ids = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("config", {}).get("architecture", "resnet34_unet") != architecture:
            raise ValueError(f"Mixed architectures in {args.fold_dir}")
        ids = checkpoint["validation_ids"]
        if set(ids) & set(checkpoint["train_ids"]):
            raise ValueError(f"Train/validation overlap in {path}")
        if not set(ids) <= images.keys():
            raise ValueError(f"Unknown validation IDs in {path}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Different validation folds in {path}")
        model.load_state_dict(checkpoint["model"])
        metrics = full_image_metrics(model, root, images, annotations, ids, device, flip4=True)
        row = {
            "checkpoint": str(path.resolve()), "epoch": checkpoint["epoch"],
            **{key: metrics[key] for key in ("active_iou", "mature_iou", "foreground_miou")},
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    baseline, selected = rows[0], max(rows, key=lambda row: row["foreground_miou"])
    summary = {"baseline": baseline, "selected": selected, "candidate_count": len(rows)}
    atomic_json(summary, args.fold_dir / "selection.json")
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    atomic_torch_save(
        {key: checkpoint[key] for key in ("epoch", "model", "fold", "train_ids", "validation_ids", "config")},
        args.fold_dir / "selected_full_miou.pt",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
