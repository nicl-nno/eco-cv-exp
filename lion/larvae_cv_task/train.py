"""Leakage-free article protocol plus the verified improvements from fold 0."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from data import IGNORE, PATCH, TrainPatches, crop_and_pad, image_path, make_mask, normalize, read_coco
from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, include_background: bool) -> torch.Tensor:
    valid = target != IGNORE
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, 3).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)
    probabilities, one_hot = logits.softmax(1) * valid, one_hot * valid
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = (probabilities + one_hot).sum((0, 2, 3))
    score = (2 * intersection + 1e-6) / (denominator + 1e-6)
    if not include_background:
        score = score[1:]
    return 1 - score.mean()


def fold_weights(images: dict, annotations: dict, ids: list[int], power: float) -> list[float]:
    """Compute weights on training IDs only: validation pixels never influence fitting."""
    pixels = np.zeros(3, dtype=np.float64)
    for image_id in ids:
        mask = make_mask(images[image_id], annotations[image_id])
        pixels += np.bincount(mask[mask != IGNORE], minlength=3)
    proportions = pixels / pixels.sum()
    weights = (1.0 / proportions) ** power
    return (weights * (3.0 / weights.sum())).tolist()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


@torch.no_grad()
def central_validation(model, root, images, annotations, ids, criterion, device) -> tuple[float, dict[str, float]]:
    model.eval()
    losses = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    for image_id in ids:
        info = images[image_id]
        image = np.asarray(Image.open(image_path(root, info)).convert("RGB"))
        mask = make_mask(info, annotations[image_id])
        y, x = max(0, (mask.shape[0] - PATCH) // 2), max(0, (mask.shape[1] - PATCH) // 2)
        image, mask = crop_and_pad(image, mask, y, x)
        tensor = normalize(image).unsqueeze(0).to(device)
        target = torch.from_numpy(mask.copy()).long().unsqueeze(0).to(device)
        logits = model(tensor)
        losses.append(float(criterion(logits, target)))
        prediction = logits.argmax(1)[0].cpu().numpy()
        valid = mask != IGNORE
        encoded = mask[valid].astype(np.int64) * 3 + prediction[valid].astype(np.int64)
        confusion += np.bincount(encoded, minlength=9).reshape(3, 3)
    ious = []
    metrics = {}
    for class_id, name in ((1, "active"), (2, "mature")):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        iou = float(tp / max(1, tp + fp + fn))
        metrics[f"central_{name}_iou"] = iou
        ious.append(iou)
    metrics["central_foreground_miou"] = float(np.mean(ious))
    return float(np.mean(losses)), metrics


@torch.no_grad()
def tiled_probabilities(model, image: np.ndarray, device: torch.device, flip4: bool = True) -> torch.Tensor:
    """Non-overlapping native-resolution tiles; Flip4 averages probabilities, not logits."""
    height, width = image.shape[:2]
    padded_h, padded_w = math.ceil(height / PATCH) * PATCH, math.ceil(width / PATCH) * PATCH
    padded = np.pad(image, ((0, padded_h - height), (0, padded_w - width), (0, 0)), mode="reflect")
    result = torch.empty((3, padded_h, padded_w), dtype=torch.float32)
    model.eval()
    for y in range(0, padded_h, PATCH):
        for x in range(0, padded_w, PATCH):
            tensor = normalize(padded[y:y + PATCH, x:x + PATCH]).unsqueeze(0).to(device)
            variants = [(tensor, ())]
            if flip4:
                variants += [
                    (torch.flip(tensor, (-1,)), (-1,)),
                    (torch.flip(tensor, (-2,)), (-2,)),
                    (torch.flip(tensor, (-2, -1)), (-2, -1)),
                ]
            probability = 0
            for transformed, inverse_axes in variants:
                current = model(transformed).softmax(1)
                probability = probability + (torch.flip(current, inverse_axes) if inverse_axes else current)
            result[:, y:y + PATCH, x:x + PATCH] = (probability / len(variants))[0].cpu()
    return result[:, :height, :width]


@torch.no_grad()
def full_image_metrics(model, root, images, annotations, ids, device, flip4: bool) -> dict[str, float]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for image_id in ids:
        info = images[image_id]
        image = np.asarray(Image.open(image_path(root, info)).convert("RGB"))
        target = make_mask(info, annotations[image_id])
        prediction = tiled_probabilities(model, image, device, flip4).argmax(0).numpy()
        valid = target != IGNORE
        encoded = target[valid].astype(np.int64) * 3 + prediction[valid].astype(np.int64)
        confusion += np.bincount(encoded, minlength=9).reshape(3, 3)
    metrics: dict[str, float] = {}
    for class_id, name in enumerate(("background", "active", "mature")):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        metrics[f"{name}_iou"] = float(tp / max(1, tp + fp + fn))
        metrics[f"{name}_dice"] = float(2 * tp / max(1, 2 * tp + fp + fn))
        metrics[f"{name}_precision"] = float(tp / max(1, tp + fp))
        metrics[f"{name}_recall"] = float(tp / max(1, tp + fn))
    metrics["foreground_miou"] = (metrics["active_iou"] + metrics["mature_iou"]) / 2
    metrics["val_miou"] = metrics["foreground_miou"]
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def resolve_path(config_file: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_file.parent / path).resolve()


def data_splits(images: dict[int, dict], config: dict) -> tuple[np.ndarray, list]:
    """Original image-ID CV, or explicit acquisition-scene CV with exclusion audit."""
    excluded = set(config.get("exclude_files", []))
    present = {Path(info["file_name"]).name for info in images.values()}
    if excluded - present:
        raise ValueError(f"Unknown excluded files: {excluded - present}")
    ids = np.array([image_id for image_id in sorted(images)
                    if Path(images[image_id]["file_name"]).name not in excluded], dtype=np.int64)
    scenes = config.get("scene_groups")
    if scenes:
        group_ids = []
        for image_id in ids:
            stem = Path(images[int(image_id)]["file_name"]).stem
            if not stem.isdigit():
                raise ValueError(f"Scene ranges require numeric filenames: {stem}")
            number = int(stem)
            matches = [row["name"] for row in scenes if int(row["first"]) <= number <= int(row["last"])]
            if len(matches) != 1:
                raise ValueError(f"Image {stem} matched {len(matches)} scene groups")
            group_ids.append(matches[0])
        if len(set(group_ids)) < int(config["n_splits"]):
            raise ValueError("Fewer distinct scenes than requested folds")
        groups = np.asarray(group_ids)
    else:
        groups = ids
    return ids, list(GroupKFold(n_splits=int(config["n_splits"])).split(ids, groups=groups))


def train_fold(config: dict, config_file: Path, fold: int) -> dict:
    root = resolve_path(config_file, config["data_root"])
    output = resolve_path(config_file, config["output_dir"]) / f"fold_{fold}"
    output.mkdir(parents=True, exist_ok=True)
    torch_home = config.get("torch_home")
    if torch_home:
        os.environ.setdefault("TORCH_HOME", str(resolve_path(config_file, torch_home)))

    images, annotations = read_coco(root)
    ids, splits = data_splits(images, config)
    train_indices, validation_indices = splits[fold]
    train_ids, validation_ids = ids[train_indices].tolist(), ids[validation_indices].tolist()
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Leakage detected: train and validation contain the same source image")
    if config.get("scene_groups"):
        image_to_scene = {int(image_id): next(row["name"] for row in config["scene_groups"]
                                   if int(row["first"]) <= int(Path(images[int(image_id)]["file_name"]).stem)
                                   <= int(row["last"])) for image_id in ids}
        if {image_to_scene[i] for i in train_ids} & {image_to_scene[i] for i in validation_ids}:
            raise RuntimeError("Leakage detected: train and validation contain the same scene")

    seed_everything(int(config["seed"]) + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config.get("architecture", "resnet34_unet"), pretrained=bool(config["pretrained"]),
    ).to(device)
    if config.get("encoder_checkpoint"):
        encoder_path = resolve_path(config_file, config["encoder_checkpoint"])
        external_encoder = torch.load(encoder_path, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(external_encoder["encoder"])
    if config.get("reset_seed_after_model_init", False):
        seed_everything(int(config["seed"]) + fold)
    weights = fold_weights(images, annotations, train_ids, float(config["class_weight_power"]))
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    def criterion(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=weight_tensor, ignore_index=IGNORE)
        dice = dice_loss(logits, target, bool(config["include_background_in_dice"]))
        return float(config["ce_weight"]) * ce + float(config["dice_weight"]) * dice

    if "encoder_learning_rate" in config:
        encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
        decoder_parameters = [parameter for parameter in model.parameters()
                              if id(parameter) not in encoder_ids]
        optimizer_parameters = [
            {"params": decoder_parameters, "lr": float(config["learning_rate"])},
            {"params": model.encoder.parameters(), "lr": float(config["encoder_learning_rate"])},
        ]
    else:
        optimizer_parameters = model.parameters()
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config["lr_factor"]), patience=int(config["lr_patience"]),
    )
    amp_enabled = bool(config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation = int(config.get("grad_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("grad_accumulation_steps must be positive")
    dataset = TrainPatches(
        root, images, annotations, train_ids,
        patches_per_image=int(config["patches_per_image"]),
        foreground_probability=float(config["foreground_probability"]),
        copy_paste_probability=float(config.get("copy_paste_probability", 0.0)),
        copy_paste_max_objects=int(config.get("copy_paste_max_objects", 2)),
        scale_range=tuple(config.get("augmentation_scale_range", (0.9, 1.1))),
        color_jitter_probability=float(config.get("color_jitter_probability", 0.0)),
        elastic_probability=float(config.get("elastic_probability", 0.3)),
    )
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
    )

    start_epoch, best_loss, best_epoch, stale, history = 1, float("inf"), 0, 0, []
    last_path, best_path = output / "last.pt", output / "best.pt"
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("python_rng_state"):
            random.setstate(checkpoint["python_rng_state"])
            np.random.set_state(checkpoint["numpy_rng_state"])
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss, best_epoch = float(checkpoint["best_loss"]), int(checkpoint["best_epoch"])
        stale, history = int(checkpoint["stale"]), checkpoint.get("history", [])

    result_path = output / "result.json"
    finished = start_epoch > int(config["epochs"]) or stale >= int(config["early_stopping_patience"])
    if finished and result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if stale >= int(config["early_stopping_patience"]):
        start_epoch = int(config["epochs"]) + 1

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        model.train()
        running, started = 0.0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(inputs), targets)
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += loss.detach().item()
        validation_loss, validation_metrics = central_validation(
            model, root, images, annotations, validation_ids, criterion, device,
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
        row = {
            "epoch": epoch, "train_loss": running / max(1, len(loader)),
            "val_loss": validation_loss, "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started, **validation_metrics,
        }
        if "encoder_learning_rate" in config:
            row["encoder_learning_rate"] = optimizer.param_groups[1]["lr"]
        history.append(row)
        state = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "best_loss": best_loss, "best_epoch": best_epoch, "stale": stale, "history": history,
            "fold": fold, "train_ids": train_ids, "validation_ids": validation_ids,
            "class_weights": weights, "config": config,
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(state, best_path)
        candidate_every = int(config.get("candidate_every", 0))
        if candidate_every > 0 and (improved or epoch % candidate_every == 0):
            candidate = {
                "epoch": epoch, "model": model.state_dict(), "fold": fold,
                "train_ids": train_ids, "validation_ids": validation_ids,
                "config": config,
            }
            atomic_torch_save(candidate, output / f"candidate_epoch_{epoch:03d}.pt")
        atomic_json({**row, "best_epoch": best_epoch, "best_val_loss": best_loss}, output / "progress.json")
        print(json.dumps({"fold": fold, **row, "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
        if stale >= int(config["early_stopping_patience"]):
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = full_image_metrics(
        model, root, images, annotations, validation_ids, device,
        flip4=config["full_image_tta"] == "flip4",
    )
    result = {
        "fold": fold, "best_epoch": checkpoint["best_epoch"], "best_val_loss": checkpoint["best_loss"],
        "class_weights": weights, "train_ids": train_ids, "validation_ids": validation_ids,
        "no_source_image_leakage": True, "metrics": metrics,
    }
    atomic_json(result, result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--fold", type=int, help="Override fold from config")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--architecture", help="Override model architecture")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()
    config_file = args.config.resolve()
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if args.architecture is not None:
        config["architecture"] = args.architecture
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    folds = range(int(config["n_splits"])) if args.all_folds else [args.fold if args.fold is not None else int(config["fold"])]
    results = [train_fold(config, config_file, fold) for fold in folds]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
