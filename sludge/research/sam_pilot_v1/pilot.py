"""Frozen, two-model FIT-only SAM feasibility experiment. Never evaluates test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import box_iou, nms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE.parent / "sam_upstream"))
sys.path.insert(0, str(HERE.parent / "efficient_sam_upstream"))

from overnight_experiments import FileLock  # noqa: E402

from sludge_micro.config import load_config  # noqa: E402
from sludge_micro.imaging import read_bgr  # noqa: E402
from sludge_micro.runs import split_samples  # noqa: E402
from sludge_micro.runtime import package_digest  # noqa: E402
from sludge_micro.split import load_split  # noqa: E402

CONFIG = (
    ROOT
    / "experiments_updated_2026_10_02/overnight_2026_10_22/queue_configs"
    / "fold_0_cosine30_s20261022.yaml"
)
LOCKS = ROOT / "outputs_updated_2026_10_02/overnight_2026_10_21"
MODELS = {
    "sam_vit_b": {
        "repo": "sam_upstream",
        "module": "segment_anything",
        "commit": "dca509fe793f601edb92606367a655c15ac00fdf",
        "weight": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
    "efficient_sam_vitt": {
        "repo": "efficient_sam_upstream",
        "module": "efficient_sam",
        "commit": "d525f622e6f640acf5a0fc37c7ca1f243da5bde0",
        "weight": "efficient_sam_vitt.pt",
        "url": "https://raw.githubusercontent.com/yformer/EfficientSAM/d525f622e6f640acf5a0fc37c7ca1f243da5bde0/weights/efficient_sam_vitt.pt",
    },
}


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)


def pin(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest}


def verify_pin(record):
    if pin(ROOT / record["path"]) != record:
        raise ValueError(f"Checksum mismatch: {record['path']}")


def select_samples(samples, manifest, count=12):
    groups = manifest.set_index("image_id")["group"].to_dict()
    ordered = sorted(
        samples, key=lambda s: hashlib.sha256(("sam-pilot-v1:" + s.image_id).encode()).hexdigest()
    )
    support = Counter(label for s in samples for label in set(s.labels))
    selected, used = [], set()
    for label in sorted(support, key=lambda k: (support[k], k)):
        if len(selected) == count:
            break
        for sample in ordered:
            if label in sample.labels and groups[sample.image_id] not in used:
                selected.append(sample)
                used.add(groups[sample.image_id])
                break
    for sample in ordered:
        if len(selected) == count:
            break
        if groups[sample.image_id] not in used:
            selected.append(sample)
            used.add(groups[sample.image_id])
    if len(selected) != count:
        raise ValueError("Insufficient independent FIT groups for registered budget")
    return selected


def prepare():
    if (HERE / "registration.json").exists():
        raise ValueError("Registration exists: never overwrite or silently re-pin")
    config = load_config(CONFIG)
    assert config.prepare.variant == "prefer_student_6"
    samples, _ = split_samples(config, "train")
    manifest = load_split(config)
    original = load_split(load_config(ROOT / "configs/overnight_2026_10_21.yaml"))
    test = original[original.split == "test"]
    selected = select_samples(samples, manifest)
    chosen = manifest[manifest.image_id.isin(s.image_id for s in selected)]
    assert set(chosen.split) == {"train"}
    heldout = manifest[manifest.split != "train"]
    outer_path = (
        ROOT / "experiments_updated_2026_10_02/grouped_cv_2026_10_21/fold_0/split_manifest.parquet"
    )
    outer_manifest = pd.read_parquet(outer_path)
    outer = outer_manifest[outer_manifest.split != "train"]
    for key in ("image_id", "group", "file_sha256", "pixel_sha256"):
        assert not set(chosen[key]) & set(test[key]), f"Original test overlap: {key}"
        assert not set(chosen[key]) & set(heldout[key]), f"Heldout overlap: {key}"
        assert not set(chosen[key]) & set(outer[key]), f"Outer overlap: {key}"
    inputs = [pin(HERE / p) for p in ("protocol.md", "pilot.py", "test_pilot.py")]
    inputs += [pin(CONFIG), pin(ROOT / "configs/overnight_2026_10_21.yaml")]
    inputs.append(pin(outer_path))
    for cfg in (config, load_config(ROOT / "configs/overnight_2026_10_21.yaml")):
        inputs.append(pin(cfg.resolve(cfg.paths.split_dir) / "split_manifest.parquet"))
    data_dir = config.resolve(config.paths.processed_dir) / "variants/prefer_student_6"
    inputs += [pin(data_dir / p) for p in ("images.parquet", "annotations.parquet")]
    models = {}
    for name, spec in MODELS.items():
        source = HERE.parent / spec["repo"]
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        assert commit == spec["commit"]
        assert not subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True
        ).strip()
        weight = HERE / "weights" / spec["weight"]
        if name == "efficient_sam_vitt":
            git_line = subprocess.check_output(
                ["git", "-C", str(source), "ls-tree", "HEAD", "weights/efficient_sam_vitt.pt"],
                text=True,
            )
            expected = git_line.split()[2]
            digest = hashlib.sha1(f"blob {weight.stat().st_size}\0".encode())
            with weight.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == expected, "EfficientSAM Git blob mismatch"
        source_pins = [pin(p) for p in sorted((source / spec["module"]).rglob("*.py"))]
        source_pins.append(pin(source / "LICENSE"))
        models[name] = {**spec, "weights_pin": pin(weight), "source_pins": source_pins}
    rows = []
    metadata = chosen.set_index("image_id")
    for s in selected:
        assert all(label > 0 for label in s.labels)
        image_pin = pin(s.path)
        assert image_pin["sha256"] == metadata.loc[s.image_id, "file_sha256"]
        rows.append(
            {
                "image_id": s.image_id,
                "image": image_pin,
                "group": metadata.loc[s.image_id, "group"],
                "width": s.width,
                "height": s.height,
                "boxes": s.boxes,
                "labels": s.labels,
            }
        )
    registration = {
        "created_at": datetime.now(UTC).isoformat(),
        "version": "sam-pilot-v1",
        "models": models,
        "inputs": inputs,
        "frames": rows,
        "classes": list(config.classes.names),
        "package_sha256": package_digest(),
        "seed": 20261023,
        "grid": 16,
        "batch": 8,
        "max_seconds": 1200,
        "quality": {
            "pred_iou": 0.88,
            "stability": 0.95,
            "stability_offset": 1.0,
            "min_area": 16,
            "nms_iou": 0.7,
            "max_boxes": 100,
        },
        "test_reopened": False,
        "heldout_used": False,
        "training": False,
    }
    write_new(HERE / "registration.json", registration)
    print(
        json.dumps(
            {
                "registered_frames": len(rows),
                "objects": sum(len(r["boxes"]) for r in rows),
                "support": dict(Counter(k for r in rows for k in r["labels"])),
            }
        )
    )


def mask_records(logits, scores):
    """One record per candidate; exclusive xyxy coordinates, no GT access."""
    if not torch.isfinite(logits).all() or not torch.isfinite(scores).all():
        raise ValueError("Nonfinite model output")
    masks = logits > 0
    areas = masks.sum((-2, -1))
    h, w = masks.shape[-2:]
    cols, rows = masks.any(-2), masks.any(-1)
    xs, ys = torch.arange(w, device=masks.device), torch.arange(h, device=masks.device)
    boxes = torch.stack(
        [
            torch.where(cols, xs, w).min(-1).values,
            torch.where(rows, ys, h).min(-1).values,
            torch.where(cols, xs + 1, 0).max(-1).values,
            torch.where(rows, ys + 1, 0).max(-1).values,
        ],
        -1,
    )
    boxes[areas == 0] = 0
    stability = (logits > 1).sum((-2, -1)) / (logits > -1).sum((-2, -1)).clamp_min(1)
    return [
        {"box": b, "score": float(s), "area": int(a), "stability": float(t)}
        for b, s, a, t in zip(
            boxes.cpu().tolist(),
            scores.cpu().tolist(),
            areas.cpu().tolist(),
            stability.cpu().tolist(),
            strict=True,
        )
    ]


def filter_records(records, quality):
    selected = [
        r
        for r in records
        if r["area"] >= 16
        and np.isfinite(r["score"])
        and (not quality or (r["score"] > 0.88 and r["stability"] >= 0.95))
    ]
    if not selected:
        return []
    boxes = torch.tensor([r["box"] for r in selected], dtype=torch.float32)
    scores = torch.tensor([r["score"] for r in selected])
    return [selected[i] for i in nms(boxes, scores, 0.7).tolist()[:100]]


def measure(records, gt, labels):
    boxes = torch.tensor([r["box"] for r in records], dtype=torch.float32).reshape(-1, 4)
    targets = torch.tensor(gt, dtype=torch.float32).reshape(-1, 4)
    ious = box_iou(boxes, targets).numpy()
    used, matches, duplicates = set(), [], 0
    for i in sorted(range(len(records)), key=lambda n: (-records[n]["score"], n)):
        choices = [j for j in range(len(gt)) if j not in used and ious[i, j] >= 0.5]
        if choices:
            j = max(choices, key=lambda k: ious[i, k])
            used.add(j)
            matches.append(j)
        elif len(gt) and (ious[i] >= 0.5).any():
            duplicates += 1
    return {
        "tp": len(matches),
        "fp": len(records) - len(matches),
        "fn": len(gt) - len(matches),
        "duplicates": duplicates,
        "predicted_count": len(records),
        "gt_count": len(gt),
        "covered_gt": int((ious >= 0.5).any(0).sum()),
        "matched_by_class": dict(Counter(labels[j] for j in matches)),
        "support_by_class": dict(Counter(labels)),
    }


class Engine:
    def __init__(self, name, spec):
        self.name = name
        weight = ROOT / spec["weights_pin"]["path"]
        if name == "sam_vit_b":
            from segment_anything import SamPredictor, sam_model_registry

            self.model = sam_model_registry["vit_b"]()
            self.model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
            self.model.eval().cuda()
            self.predictor = SamPredictor(self.model)
        else:
            from efficient_sam.efficient_sam import build_efficient_sam

            self.model = build_efficient_sam(192, 3)
            self.model.load_state_dict(
                torch.load(weight, map_location="cpu", weights_only=True)["model"]
            )
            self.model.eval().cuda()

    def set_image(self, rgb):
        self.h, self.w = rgb.shape[:2]
        if self.name == "sam_vit_b":
            self.predictor.set_image(rgb)
        else:
            image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().cuda()[None] / 255
            self.embedding = self.model.get_image_embeddings(image)

    def predict(self, coordinates, is_box=False):
        coords = torch.tensor(coordinates, dtype=torch.float32, device="cuda")
        if self.name == "sam_vit_b":
            kwargs = {"point_coords": None, "point_labels": None}
            if is_box:
                kwargs["boxes"] = self.predictor.transform.apply_boxes_torch(
                    coords, (self.h, self.w)
                )
            else:
                kwargs["point_coords"] = self.predictor.transform.apply_coords_torch(
                    coords, (self.h, self.w)
                )[:, None]
                kwargs["point_labels"] = torch.ones(
                    (len(coords), 1), dtype=torch.long, device="cuda"
                )
            logits, scores, _ = self.predictor.predict_torch(
                **kwargs, multimask_output=True, return_logits=True
            )
        else:
            points = coords.reshape(1, -1, 2 if is_box else 1, 2)
            labels = (
                torch.tensor([2, 3], device="cuda").expand(1, len(coords), 2)
                if is_box
                else torch.ones((1, len(coords), 1), dtype=torch.long, device="cuda")
            )
            logits, scores = self.model.predict_masks(
                self.embedding, points, labels, True, self.h, self.w, self.h, self.w
            )
            logits, scores = logits[0], scores[0]
        return logits, scores


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def overlay(rgb, frame, records, path):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for box in frame["boxes"]:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (60, 210, 60), 2)
    for record in records:
        x1, y1, x2, y2 = map(int, record["box"])
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 140, 255), 1)
    cv2.putText(
        bgr,
        "GT green / automatic proposals orange",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok
    with path.open("xb") as stream:
        stream.write(encoded.tobytes())


def run():
    with FileLock(LOCKS / "queue.lock"), FileLock(LOCKS / "gpu.lock"):
        reg = json.loads((HERE / "registration.json").read_text(encoding="utf-8"))
        assert reg["package_sha256"] == package_digest()
        for record in reg["inputs"]:
            verify_pin(record)
        for spec in reg["models"].values():
            for record in [spec["weights_pin"], *spec["source_pins"]]:
                verify_pin(record)
        for frame in reg["frames"]:
            verify_pin(frame["image"])
        write_new(
            HERE / "run_started.json",
            {"at": datetime.now(UTC).isoformat(), "registration": pin(HERE / "registration.json")},
        )
        started = time.perf_counter()

        def check_budget():
            if time.perf_counter() - started >= reg["max_seconds"]:
                raise TimeoutError("Registered inference wall budget reached")

        torch.manual_seed(reg["seed"])
        torch.set_num_threads(4)
        output = HERE / "outputs"
        output.mkdir(exist_ok=False)
        all_results = []
        try:
            with torch.inference_mode():
                for name, spec in reg["models"].items():
                    check_budget()
                    engine = Engine(name, spec)
                    torch.cuda.reset_peak_memory_stats()
                    parameters = sum(p.numel() for p in engine.model.parameters())
                    for index, frame in enumerate(reg["frames"]):
                        check_budget()
                        rgb = cv2.cvtColor(
                            read_bgr(ROOT / frame["image"]["path"]), cv2.COLOR_BGR2RGB
                        )
                        t0 = sync_time()
                        engine.set_image(rgb)
                        embedding_seconds = sync_time() - t0
                        h, w = rgb.shape[:2]
                        axis = (np.arange(16) + 0.5) / 16
                        xs, ys = np.meshgrid(axis * w, axis * h)
                        grid = np.stack([xs.ravel(), ys.ravel()], -1).tolist()
                        records, oracle, times = [], {}, {}
                        t0 = sync_time()
                        for offset in range(0, len(grid), 8):
                            check_budget()
                            logits, scores = engine.predict(grid[offset : offset + 8])
                            records.extend(mask_records(logits.flatten(0, 1), scores.flatten()))
                            del logits, scores
                        times["automatic_decoder_seconds"] = sync_time() - t0
                        automatic = {
                            "quality": filter_records(records, True),
                            "raw_top100": filter_records(records, False),
                        }
                        for mode in ("gt_center", "gt_box"):
                            prompts = (
                                frame["boxes"]
                                if mode == "gt_box"
                                else [
                                    [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in frame["boxes"]
                                ]
                            )
                            chosen = []
                            t0 = sync_time()
                            for offset in range(0, len(prompts), 8):
                                check_budget()
                                logits, scores = engine.predict(
                                    prompts[offset : offset + 8], mode == "gt_box"
                                )
                                best = scores.argmax(1)
                                arange = torch.arange(len(best), device=scores.device)
                                chosen.extend(
                                    mask_records(logits[arange, best], scores[arange, best])
                                )
                                del logits, scores
                            times[mode + "_decoder_seconds"] = sync_time() - t0
                            boxes = torch.tensor(
                                [r["box"] for r in chosen], dtype=torch.float32
                            ).reshape(-1, 4)
                            gt = torch.tensor(frame["boxes"], dtype=torch.float32).reshape(-1, 4)
                            ious = box_iou(boxes, gt).diag().nan_to_num(0).tolist()
                            oracle[mode] = {"records": chosen, "prompt_box_consistency_iou": ious}
                        result = {
                            "model": name,
                            "image_id": frame["image_id"],
                            "parameters": parameters,
                            "embedding_seconds": embedding_seconds,
                            "times": times,
                            "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                            "automatic": {
                                mode: {
                                    "records": rs,
                                    "metrics": measure(rs, frame["boxes"], frame["labels"]),
                                }
                                for mode, rs in automatic.items()
                            },
                            "oracle": oracle,
                        }
                        write_new(output / f"{name}_{index:02d}.json", result)
                        if index < 2:
                            overlay(
                                rgb, frame, automatic["quality"], output / f"{name}_{index:02d}.png"
                            )
                        all_results.append(result)
                        print(f"{name} {index+1}/12 finished", flush=True)
                    del engine
                    gc.collect()
                    torch.cuda.empty_cache()
            assert reg["package_sha256"] == package_digest(), "Package changed during pilot"
            write_new(
                HERE / "results.json",
                {
                    "completed_at": datetime.now(UTC).isoformat(),
                    "package_unchanged": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "test_reopened": False,
                    "results": all_results,
                },
            )
        except Exception as exc:
            write_new(
                HERE / "failed.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "completed_frames": len(all_results),
                },
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error("Choose exactly one of --prepare or --run")
    prepare() if args.prepare else run()
