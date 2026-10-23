"""Find near-identical source photographs and cross-fold scene leakage."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from data import image_path, read_coco
from train import atomic_json


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "task" / "coco_export"


def thumbnail(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        # Fit the whole frame; orientation and camera framing are informative.
        return np.asarray(image.convert("L").resize((128, 128)), dtype=np.float32)


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    # Compensate small hand-held translations by checking a few local offsets.
    left = left[8:-8, 8:-8]
    left = left - left.mean()
    left /= max(np.linalg.norm(left), 1e-6)
    best = -1.0
    for k in range(4):
        rotated = np.rot90(right, k)
        for dy in (-4, 0, 4):
            for dx in (-4, 0, 4):
                candidate = np.roll(rotated, (dy, dx), axis=(0, 1))[8:-8, 8:-8]
                candidate = candidate - candidate.mean()
                candidate /= max(np.linalg.norm(candidate), 1e-6)
                best = max(best, float((left * candidate).sum()))
    return best


def main() -> None:
    images, annotations = read_coco(ROOT)
    ids = sorted(images)
    thumbs = {image_id: thumbnail(image_path(ROOT, images[image_id])) for image_id in ids}
    fold_ids = {}
    for fold in range(5):
        checkpoint = HERE / "runs" / "night" / "resnet34_unet" / f"fold_{fold}" / "result.json"
        validation = json.loads(checkpoint.read_text(encoding="utf-8"))["validation_ids"]
        fold_ids.update({image_id: fold for image_id in validation})
    pairs = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            score = similarity(thumbs[left], thumbs[right])
            pairs.append({"left": left, "right": right, "similarity": score,
                          "cross_fold": fold_ids[left] != fold_ids[right]})
    pairs.sort(key=lambda row: row["similarity"], reverse=True)
    hashes = {}
    for image_id in ids:
        digest = hashlib.sha256(image_path(ROOT, images[image_id]).read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(image_id)
    exact_groups = [group for group in hashes.values() if len(group) > 1]
    exact_duplicates = [
        {"ids": group, "files": [images[image_id]["file_name"] for image_id in group],
         "folds": [fold_ids[image_id] for image_id in group],
         "annotations": [{str(category): sum(int(a["category_id"]) == category
                                          for a in annotations[image_id])
                          for category in (1, 2, 3)} for image_id in group]}
        for group in exact_groups
    ]
    summary = {"top_pairs": pairs[:60], "thresholds": {
        str(t): {"pairs": sum(row["similarity"] >= t for row in pairs),
                 "cross_fold_pairs": sum(row["similarity"] >= t and row["cross_fold"] for row in pairs)}
        for t in (0.95, 0.90, 0.85, 0.80)}, "exact_duplicates": exact_duplicates}
    destination = HERE / "runs" / "hypotheses" / "source_similarity.json"
    atomic_json(summary, destination)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
