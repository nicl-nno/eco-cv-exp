"""Compare the original non-overlapping inference with blended overlapping tiles.

This evaluates an existing fold checkpoint only; it never fits on validation data.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from data import IGNORE, PATCH, image_path, make_mask, normalize, read_coco
from model import build_model
from train import atomic_json, full_image_metrics, resolve_path


def tile_starts(length: int, stride: int) -> list[int]:
    if length <= PATCH:
        return [0]
    starts = list(range(0, length - PATCH + 1, stride))
    if starts[-1] != length - PATCH:
        starts.append(length - PATCH)
    return starts


@torch.inference_mode()
def blended_probabilities(
    model, image: np.ndarray, device: torch.device, stride: int, tta: str = "flip4",
) -> torch.Tensor:
    height, width = image.shape[:2]
    padded = np.pad(
        image,
        ((0, max(0, PATCH - height)), (0, max(0, PATCH - width)), (0, 0)),
        mode="reflect",
    )
    padded_h, padded_w = padded.shape[:2]
    probabilities = torch.zeros((3, padded_h, padded_w), dtype=torch.float32)
    coverage = torch.zeros((padded_h, padded_w), dtype=torch.float32)
    # A nonzero edge weight also makes tiles at image boundaries well-defined.
    taper = np.maximum(np.hanning(PATCH).astype(np.float32), 0.05)
    weight = torch.from_numpy(np.outer(taper, taper).copy())
    for y in tile_starts(padded_h, stride):
        for x in tile_starts(padded_w, stride):
            tensor = normalize(padded[y:y + PATCH, x:x + PATCH]).unsqueeze(0).to(device)
            if tta == "flip4":
                variants = torch.cat(
                    (tensor, tensor.flip((-1,)), tensor.flip((-2,)), tensor.flip((-2, -1))),
                    dim=0,
                )
                predicted = model(variants).softmax(1)
                averaged = (
                    predicted[0] + predicted[1].flip((-1,))
                    + predicted[2].flip((-2,)) + predicted[3].flip((-2, -1))
                ).div(4).cpu()
            elif tta == "dihedral8":
                variants = torch.cat(
                    [torch.rot90(tensor, k, dims=(-2, -1)) for k in range(4)]
                    + [torch.rot90(tensor.flip((-1,)), k, dims=(-2, -1)) for k in range(4)],
                    dim=0,
                )
                predicted = model(variants).softmax(1)
                unrotated = [torch.rot90(predicted[k], -k, dims=(-2, -1)) for k in range(4)]
                unrotated += [
                    torch.rot90(predicted[4 + k], -k, dims=(-2, -1)).flip((-1,))
                    for k in range(4)
                ]
                averaged = torch.stack(unrotated).mean(0).cpu()
            else:
                raise ValueError(f"Unknown TTA mode: {tta}")
            probabilities[:, y:y + PATCH, x:x + PATCH] += averaged * weight
            coverage[y:y + PATCH, x:x + PATCH] += weight
    return (probabilities / coverage.unsqueeze(0))[:, :height, :width]


def evaluate(model, root, images, annotations, ids, device, stride: int, tta: str) -> dict:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for image_id in ids:
        info = images[image_id]
        image = np.asarray(Image.open(image_path(root, info)).convert("RGB"))
        target = make_mask(info, annotations[image_id])
        prediction = blended_probabilities(model, image, device, stride, tta).argmax(0).numpy()
        valid = target != IGNORE
        confusion += np.bincount(
            target[valid].astype(np.int64) * 3 + prediction[valid].astype(np.int64),
            minlength=9,
        ).reshape(3, 3)
    metrics = {}
    for class_id, name in ((1, "active"), (2, "mature")):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        metrics[f"{name}_iou"] = float(tp / max(1, tp + fp + fn))
    metrics["foreground_miou"] = (metrics["active_iou"] + metrics["mature_iou"]) / 2
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--tta", choices=("flip4", "dihedral8"), default="flip4")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    if not 1 <= args.stride <= PATCH:
        parser.error(f"stride must be between 1 and {PATCH}")
    config_file = args.config.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    root = resolve_path(config_file, config["data_root"])
    images, annotations = read_coco(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ids = checkpoint["validation_ids"]
    if set(ids) & set(checkpoint["train_ids"]):
        raise ValueError("Checkpoint has overlapping train/validation IDs")
    if not set(ids) <= images.keys():
        raise ValueError("Validation IDs do not match the dataset")
    architecture = checkpoint.get("config", {}).get("architecture", "resnet34_unet")
    model = build_model(architecture, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    started = time.perf_counter()
    baseline = full_image_metrics(model, root, images, annotations, ids, device, flip4=True)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    overlap = evaluate(model, root, images, annotations, ids, device, args.stride, args.tta)
    overlap_seconds = time.perf_counter() - started
    result = {
        "epoch": checkpoint["epoch"],
        "baseline": {key: baseline[key] for key in ("active_iou", "mature_iou", "foreground_miou")},
        "baseline_seconds": baseline_seconds,
        "overlap_stride": args.stride,
        "overlap_tta": args.tta,
        "overlap": overlap,
        "overlap_seconds": overlap_seconds,
    }
    if args.output:
        atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
