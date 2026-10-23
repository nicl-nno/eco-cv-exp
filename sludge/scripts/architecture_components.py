"""Isolated third-night components; never load test or mutate prepared labels."""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import replace

import cv2
import numpy as np
import overnight_calibrated as old
import overnight_experiments as night
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import resnet50
from torchvision.ops import batched_nms

from sludge_micro.checkpoint import load_restricted, state_dict_of
from sludge_micro.dataset import make_loader
from sludge_micro.geometry import ioa_matrix, iou_matrix
from sludge_micro.imaging import read_bgr
from sludge_micro.infer import PREDICTION_COLUMNS, Predictor
from sludge_micro.model import load_trained, save_checkpoint
from sludge_micro.reporting import read_json, write_parquet
from sludge_micro.runs import predict_samples, references
from sludge_micro.runtime import apply_runtime
from sludge_micro.train import (
    batchnorm_check,
    batchnorm_statistics,
    build_for_training,
    fit,
    restore_resume_state,
    save_resume_state,
    set_training_mode,
)

BOX = ["x1", "y1", "x2", "y2"]
LOG = logging.getLogger(__name__)


def check_deadline(deadline):
    if night.now() >= deadline:
        raise RuntimeError("Deadline: no more model steps")


def release_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mapping(names, component):
    if component == "agnostic":
        return dict.fromkeys(range(1, len(names) + 1), 1)
    selected = [
        i
        for i, n in enumerate(names, 1)
        if (n == "filamentous_bacteria") == (component == "filamentous")
    ]
    if component not in {"filamentous", "others"}:
        raise ValueError("Unknown detector component")
    return {label: i for i, label in enumerate(selected, 1)}


def remap_samples(samples, labels):
    if any(label <= 0 for s in samples for label in s.labels):
        raise ValueError("FIT must exclude ignore before label remapping")
    return [
        replace(
            s,
            boxes=tuple(b for b, label in zip(s.boxes, s.labels, strict=True) if label in labels),
            labels=tuple(labels[label] for label in s.labels if label in labels),
        )
        for s in samples
    ]


def completed(directory):
    path = directory / "complete.json"
    if not path.exists():
        return False
    record = read_json(path)
    for item in record["artifacts"]:
        old.verify_pin(item)
    return True


def fit_detector(config, samples, directory, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    if completed(directory):
        return
    if night.now() >= deadline:
        raise RuntimeError("Deadline: no component training")
    generator = apply_runtime(config.runtime)
    model, initialization = build_for_training(config)
    start = batchnorm_statistics(model)
    device = torch.device(config.runtime.device)
    model.to(device)
    # Also enforce the cutoff if the supervisor unexpectedly disappears.
    model.register_forward_pre_hook(lambda _model, _inputs: check_deadline(deadline))
    resume_path = directory / "resume_state.pt"
    state = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path.exists()
        else None
    )
    loader = make_loader(samples, config, True, generator)
    history = fit(
        model,
        loader,
        config,
        device,
        (directory / "train_log.jsonl", resume_path),
        generator,
        state,
    )
    if len(history) != config.train.epochs or not all(math.isfinite(r["loss"]) for r in history):
        raise ValueError("Incomplete/nonfinite detector training")
    check = batchnorm_check(model, start, config.train.batchnorm)
    if check["changed"]:
        raise ValueError("Frozen BN statistics changed")
    temporary = directory / "model.pt.part"
    save_checkpoint(model, temporary, config, {"train_ids": [s.image_id for s in samples]})
    os.replace(temporary, directory / "model.pt")
    night.atomic_json(
        directory / "complete.json",
        {
            "artifacts": [old.pin(directory / "model.pt")],
            "history": history,
            "initialization": initialization,
            "batchnorm": check,
            "images": len(samples),
            "objects": sum(len(s.labels) for s in samples),
            "train_ids": [s.image_id for s in samples],
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    del model, loader, state
    release_gpu()


def atomic_parquet(path, frame):
    temporary = path.with_name(path.name + ".part")
    write_parquet(temporary, frame)
    os.replace(temporary, path)


def detector_predictions(config, directory, samples, output):
    if output.exists() and output.with_suffix(".timing.json").exists():
        return pd.read_parquet(output)
    started = time.perf_counter()
    model, _ = load_trained(directory / "model.pt", config)
    predictor = Predictor(model, config.classes.names, 0.0, torch.device(config.runtime.device))
    predictions = predict_samples(predictor, samples)
    atomic_parquet(output, predictions)
    night.atomic_json(
        output.with_suffix(".timing.json"),
        {
            "seconds": time.perf_counter() - started,
            "images": len(samples),
            "proposals": len(predictions),
            "includes_model_load": True,
        },
    )
    del predictor, model
    release_gpu()
    return predictions


def crop_manifest(samples, predictions, seed):
    """Input samples must be usable FIT, verified by caller against the locked split."""
    ids = {s.image_id for s in samples}
    if len(ids) != len(samples) or not set(predictions.image_id) <= ids:
        raise ValueError("FIT crop contamination or duplicate IDs")
    if any(label <= 0 for s in samples for label in s.labels):
        raise ValueError("Crop FIT contains ignore")
    rng, rows = np.random.default_rng(seed), []
    for s in samples:
        boxes = np.asarray(s.boxes, float).reshape(-1, 4)

        def add(box, label, source, gt_index=None, image_id=s.image_id):
            rows.append(
                {
                    "image_id": image_id,
                    **dict(zip(BOX, map(float, box), strict=True)),
                    "label": int(label),
                    "source": source,
                    "gt_index": gt_index,
                }
            )

        for i, (box, label) in enumerate(zip(boxes, s.labels, strict=True)):
            add(box, label, "gt", i)
            scale = np.tile(box[2:] - box[:2], 2)
            jitter = box + rng.uniform(-0.1, 0.1, 4) * scale
            jitter = np.clip(jitter, [0, 0, 0, 0], [s.width, s.height, s.width, s.height])
            if np.all(jitter[2:] > jitter[:2]) and iou_matrix(jitter[None], box[None])[0, 0] >= 0.7:
                add(jitter, label, "jitter", i)
        pred = predictions[predictions.image_id == s.image_id].sort_values(
            "score", ascending=False, kind="stable"
        )
        positives, negatives = Counter(), 0
        for row in pred.itertuples():
            box = np.array([row.x1, row.y1, row.x2, row.y2])
            overlaps = iou_matrix(box[None], boxes)[0] if len(boxes) else np.array([])
            best = int(overlaps.argmax()) if len(overlaps) else None
            value = float(overlaps[best]) if best is not None else 0.0
            if value >= 0.5 and positives[best] < 2:
                add(box, s.labels[best], "fit_proposal_positive", best)
                positives[best] += 1
            elif value < 0.1 and negatives < 5:
                add(box, 0, "fit_proposal_negative")
                negatives += 1
    return pd.DataFrame(rows, columns=["image_id", *BOX, "label", "source", "gt_index"])


def crop_tensor(image, box):
    """RGB, 10% context, zero-padded square; no aspect distortion."""
    x1, y1, x2, y2 = map(float, box)
    h, w = image.shape[:2]
    dx, dy = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    left, top = max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy))
    right, bottom = min(w, math.ceil(x2 + dx)), min(h, math.ceil(y2 + dy))
    part = image[top:bottom, left:right, ::-1]
    if part.size == 0:
        raise ValueError("Empty crop")
    side = max(part.shape[:2])
    square = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - part.shape[0]) // 2, (side - part.shape[1]) // 2
    square[oy : oy + part.shape[0], ox : ox + part.shape[1]] = part
    rgb = cv2.resize(square, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def normalize(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (batch - mean) / std


class CropDataset(Dataset):
    def __init__(self, samples, manifest, seed):
        self.rows = manifest.to_dict("records")
        self.paths = {s.image_id: s.path for s in samples}
        if not set(manifest.image_id) <= set(self.paths):
            raise ValueError("Crop sample contamination")
        self.seed, self.epoch = seed, 0
        self.cache = {}  # cropped uint8 tensors, not full-frame images

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if index not in self.cache:
            image = read_bgr(self.paths[row["image_id"]])
            self.cache[index] = (crop_tensor(image, [row[k] for k in BOX]) * 255).round().byte()
        tensor = self.cache[index].float() / 255
        rng = np.random.default_rng([self.seed, self.epoch, index])
        tensor = torch.rot90(tensor, int(rng.integers(4)), (1, 2))
        if rng.random() < 0.5:
            tensor = tensor.flip(1)
        if rng.random() < 0.5:
            tensor = tensor.flip(2)
        return tensor, row["label"]


def build_classifier(pretrained, classes):
    model = resnet50(weights=None, num_classes=classes + 1)
    state = state_dict_of(load_restricted(pretrained), pretrained)
    prefix = "backbone.body."
    body = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    loaded = model.load_state_dict(body, strict=False)
    if set(loaded.missing_keys) != {"fc.weight", "fc.bias"} or loaded.unexpected_keys:
        raise ValueError(f"COCO classifier backbone mismatch: {loaded}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("layer4.", "fc.")))
    return model


def crop_weights(labels):
    counts = Counter(labels)
    return [min(3.0, math.sqrt(max(counts.values()) / counts[label])) for label in labels]


def fit_classifier(config, samples, manifest, directory, deadline):
    directory.mkdir(parents=True, exist_ok=True)
    if completed(directory):
        return
    if night.now() >= deadline:
        raise RuntimeError("Deadline: no classifier training")
    generator = apply_runtime(config.runtime)
    model = build_classifier(
        config.resolve(config.paths.pretrained_weights), len(config.classes.names)
    )
    start = batchnorm_statistics(model)
    device = torch.device(config.runtime.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    dataset = CropDataset(samples, manifest, config.runtime.seed)
    sampler = WeightedRandomSampler(
        crop_weights(manifest.label.tolist()), len(manifest), True, generator=generator
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0, generator=generator)
    resume_path = directory / "resume_state.pt"
    state = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path.exists()
        else None
    )
    history = restore_resume_state(state, model, optimizer, generator) if state else []
    with (directory / "train_log.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(len(history), 20):
            if night.now() >= deadline:
                raise RuntimeError("Deadline: classifier checkpoint saved at last epoch")
            started = time.perf_counter()
            dataset.epoch = epoch
            set_training_mode(model, "frozen")
            lr = 1e-4 * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / 19)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            total, seen = 0.0, 0
            for tensors, labels in loader:
                check_deadline(deadline)
                optimizer.zero_grad(set_to_none=True)
                logits = model(normalize(tensors.to(device)))
                loss = torch.nn.functional.cross_entropy(logits, labels.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite classifier loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(labels)
                seen += len(labels)
            history.append({"epoch": epoch + 1, "loss": total / seen, "lr": lr})
            save_resume_state(resume_path, model, optimizer, generator, history)
            log.write(json.dumps({**history[-1], "seconds": time.perf_counter() - started}) + "\n")
            log.flush()
            LOG.info("classifier epoch %d loss %.4f", epoch + 1, total / seen)
    check = batchnorm_check(model, start, "frozen")
    if check["changed"]:
        raise ValueError("Classifier BN changed")
    temporary = directory / "model.pt.part"
    torch.save(
        {
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "classes": ["background", *config.classes.names],
            "architecture": "resnet50",
        },
        temporary,
    )
    os.replace(temporary, directory / "model.pt")
    night.atomic_json(
        directory / "complete.json",
        {
            "artifacts": [old.pin(directory / "model.pt"), old.pin(directory / "crops.parquet")],
            "history": history,
            "batchnorm": check,
            "crop_counts": {str(k): int(v) for k, v in Counter(manifest.label).items()},
            "train_ids": sorted(set(manifest.image_id)),
            "sampling": "sqrt(max_class_count/class_count)_cap3",
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    del model, optimizer, loader, dataset, state
    release_gpu()


def cascade_rows(image_id, proposals, probabilities, names):
    if len(proposals) != len(probabilities):
        raise ValueError("Proposal/classifier shape mismatch")
    if not len(proposals):
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    labels = probabilities.argmax(1)
    scores = probabilities[np.arange(len(labels)), labels] * proposals.score.to_numpy(float)
    keep = labels > 0
    boxes = torch.tensor(proposals[BOX].to_numpy()[keep], dtype=torch.float32)
    selected = batched_nms(
        boxes, torch.tensor(scores[keep], dtype=torch.float32), torch.tensor(labels[keep]), 0.5
    )[:100].numpy()
    indices = np.flatnonzero(keep)[selected]
    result = proposals.iloc[indices].copy()
    result["image_id"] = image_id
    result["class_name"] = [names[labels[i] - 1] for i in indices]
    result["score"] = scores[indices]
    return result[PREDICTION_COLUMNS].reset_index(drop=True)


def classify_proposals(config, directory, samples, proposals, output):
    if output.exists() and output.with_suffix(".timing.json").exists():
        return pd.read_parquet(output)
    started = time.perf_counter()
    payload = load_restricted(directory / "model.pt")
    if payload["classes"] != ["background", *config.classes.names]:
        raise ValueError("Classifier class mapping changed")
    model = resnet50(weights=None, num_classes=len(config.classes.names) + 1)
    model.load_state_dict(payload["model"])
    device = torch.device(config.runtime.device)
    model.to(device).eval()
    frames, oracle = [], []
    with torch.inference_mode():
        for sample in samples:
            part = proposals[proposals.image_id == sample.image_id].reset_index(drop=True)
            image = read_bgr(sample.path)
            probs = []
            for begin in range(0, len(part), 32):
                boxes = part.iloc[begin : begin + 32][BOX].to_numpy()
                tensors = torch.stack([crop_tensor(image, b) for b in boxes]).to(device)
                probs.extend(model(normalize(tensors)).softmax(1).cpu().numpy())
            frames.append(
                cascade_rows(sample.image_id, part, np.asarray(probs), config.classes.names)
            )
            # Oracle diagnostic only for outer validation; never part of optimization.
            if "outer" in output.name:
                targets = [
                    (b, label)
                    for b, label in zip(sample.boxes, sample.labels, strict=True)
                    if label > 0
                ]
                for begin in range(0, len(targets), 32):
                    group = targets[begin : begin + 32]
                    tensors = torch.stack([crop_tensor(image, b) for b, _ in group]).to(device)
                    labels = model(normalize(tensors)).argmax(1).cpu().tolist()
                    oracle.extend(
                        {"image_id": sample.image_id, "truth": int(label), "prediction": int(p)}
                        for (_, label), p in zip(group, labels, strict=True)
                    )
    result = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    atomic_parquet(output, result)
    confusion = np.zeros((len(config.classes.names) + 1,) * 2, dtype=int)
    for row in oracle:
        confusion[row["truth"], row["prediction"]] += 1
    night.atomic_json(
        output.with_suffix(".timing.json"),
        {
            "seconds": time.perf_counter() - started,
            "images": len(samples),
            "proposals": len(proposals),
            "retained": len(result),
            "oracle_gt_crops": len(oracle),
            "oracle_confusion": confusion.tolist(),
            "oracle_accuracy": (
                float(np.trace(confusion) / confusion.sum()) if confusion.sum() else None
            ),
            "oracle_class_order": ["background", *config.classes.names],
            "includes_oracle_and_model_load": True,
        },
    )
    del model, payload
    release_gpu()
    return result


def merge_specialists(first, second, suppress=True, return_counts=False):
    combined = pd.concat([first.assign(source=0), second.assign(source=1)], ignore_index=True)
    rows = []
    suppressed, capped = 0, 0
    for _, part in combined.groupby("image_id", sort=True):
        part = part.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
        boxes, sources = part[BOX].to_numpy(float), part.source.to_numpy()
        kept = []
        for i in range(len(part)):
            if suppress and kept:
                cross = [j for j in kept if sources[j] != sources[i]]
                if cross and np.any(iou_matrix(boxes[i : i + 1], boxes[cross]) > 0.8):
                    suppressed += 1
                    continue
            kept.append(i)
        capped += max(0, len(kept) - 100)
        rows.append(part.iloc[kept[:100]][PREDICTION_COLUMNS])
    result = (
        pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=PREDICTION_COLUMNS)
    )
    if return_counts:
        return result, {
            "input": len(combined),
            "cross_suppressed": suppressed,
            "top100_discarded": capped,
            "output": len(result),
        }
    return result


def diagnostics(config, predictions, anns):
    """Class-agnostic greedy localization, then class accuracy; honor ignore areas."""
    target, ignore = references(anns)
    predictions = predictions[predictions.score >= 0.05]
    tp = fp = ignored = duplicates = correct = 0
    for image_id in sorted(set(target.image_id) | set(predictions.image_id)):
        pred = predictions[predictions.image_id == image_id].sort_values(
            "score", ascending=False, kind="stable"
        )
        gt = target[target.image_id == image_id]
        ig = ignore[ignore.image_id == image_id]
        overlap = iou_matrix(pred[BOX].to_numpy(float), gt[BOX].to_numpy(float))
        cover = ioa_matrix(pred[BOX].to_numpy(float), ig[BOX].to_numpy(float))
        used = set()
        for i, row in enumerate(pred.itertuples()):
            eligible = [j for j in range(len(gt)) if j not in used and overlap[i, j] >= 0.5]
            if eligible:
                best = max(eligible, key=lambda j: overlap[i, j])
                used.add(best)
                tp += 1
                correct += int(row.class_name == gt.iloc[best].class_name)
            elif len(ig) and cover[i].max() >= config.evaluation.ignore_ioa_threshold:
                ignored += 1
            else:
                fp += 1
                duplicates += int(any(overlap[i, j] >= 0.5 for j in used))
    return {
        "fixed_score": 0.05,
        "localization_tp": tp,
        "localization_fp": fp,
        "localization_fn": len(target) - tp,
        "ignored": ignored,
        "duplicate_fp": duplicates,
        "localization_precision": tp / (tp + fp) if tp + fp else None,
        "localization_recall": tp / len(target) if len(target) else None,
        "matched_class_correct": correct,
        "matched_class_accuracy": correct / tp if tp else None,
    }
