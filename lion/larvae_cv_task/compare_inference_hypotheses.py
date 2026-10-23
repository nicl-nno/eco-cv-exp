"""Test a fixed 1:1 U-Net/SE ensemble and SE Dihedral8 on identical folds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import IGNORE, image_path, make_mask, read_coco
from evaluate_tiling import blended_probabilities
from model import build_model
from train import atomic_json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "task" / "coco_export"
RUNS = HERE / "runs" / "night"
OUTPUT = HERE / "runs" / "hypotheses"
BASELINE = [0.6099695964166286, 0.5356911211913897,
            0.5717167764054192, 0.539089073039853]


def load_model(name: str, fold: int, device: torch.device):
    path = RUNS / name / f"fold_{fold}" / "selected_full_miou.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["config"]["architecture"] == name
    model = build_model(name, pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def confusion(prediction: torch.Tensor, target: np.ndarray) -> np.ndarray:
    valid = target != IGNORE
    encoded = target[valid].astype(np.int64) * 3 + prediction.argmax(0).numpy()[valid]
    return np.bincount(encoded, minlength=9).reshape(3, 3)


def metrics(matrix: np.ndarray) -> dict[str, float]:
    result = {}
    for index, name in ((1, "active"), (2, "mature")):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        result[f"{name}_iou"] = float(tp / max(1, tp + fp + fn))
    result["foreground_miou"] = (result["active_iou"] + result["mature_iou"]) / 2
    return result


def evaluate_fold(fold: int, images: dict, annotations: dict, device: torch.device) -> dict:
    se, se_checkpoint = load_model("resnet34_unet_se", fold, device)
    plain, plain_checkpoint = load_model("resnet34_unet", fold, device)
    ids = se_checkpoint["validation_ids"]
    if ids != plain_checkpoint["validation_ids"] or set(ids) & set(se_checkpoint["train_ids"]):
        raise ValueError(f"Validation mismatch or leakage on fold {fold}")
    modes = ("se_flip4", "unet_flip4", "ensemble_flip4", "se_dihedral8")
    matrices = {mode: np.zeros((3, 3), dtype=np.int64) for mode in modes}
    seconds = {mode: 0.0 for mode in modes}
    per_image = []
    for image_id in ids:
        info = images[image_id]
        image = np.asarray(Image.open(image_path(ROOT, info)).convert("RGB"))
        target = make_mask(info, annotations[image_id])
        started = time.perf_counter()
        se_prob = blended_probabilities(se, image, device, 448, "flip4")
        seconds["se_flip4"] += time.perf_counter() - started
        started = time.perf_counter()
        plain_prob = blended_probabilities(plain, image, device, 448, "flip4")
        seconds["unet_flip4"] += time.perf_counter() - started
        started = time.perf_counter()
        dihedral_prob = blended_probabilities(se, image, device, 448, "dihedral8")
        seconds["se_dihedral8"] += time.perf_counter() - started
        seconds["ensemble_flip4"] = seconds["se_flip4"] + seconds["unet_flip4"]
        probabilities = {"se_flip4": se_prob, "unet_flip4": plain_prob,
                         "ensemble_flip4": (se_prob + plain_prob) / 2,
                         "se_dihedral8": dihedral_prob}
        row = {"image_id": image_id}
        for name, probability in probabilities.items():
            matrix = confusion(probability, target)
            matrices[name] += matrix
            row[name] = metrics(matrix)["foreground_miou"]
        per_image.append(row)
        print(json.dumps({"fold": fold, **row}), flush=True)
    results = {name: {**metrics(matrix), "seconds": seconds[name]}
               for name, matrix in matrices.items()}
    if abs(results["se_flip4"]["foreground_miou"] - BASELINE[fold]) > 1e-7:
        raise RuntimeError(f"SE baseline does not reproduce on fold {fold}")
    return {"fold": fold, "validation_ids": ids, "results": results,
            "per_image": per_image}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    images, annotations = read_coco(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        if not 0 <= fold <= 3:
            parser.error("Only folds 0–3 have both checkpoints")
        result = evaluate_fold(fold, images, annotations, device)
        atomic_json(result, OUTPUT / f"inference_fold_{fold}.json")
        print(json.dumps({"fold": fold, "results": result["results"]}), flush=True)


if __name__ == "__main__":
    main()
