"""Dataset imaging statistics (brightness, contrast, entropy, sharpness, …)."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from maa.data.zenodo import SampleRecord, foreground_fraction, load_manifest, scan_zenodo_dataset

logger = logging.getLogger(__name__)


@dataclass
class ClassStats:
    class_name: str
    count: int
    mean_width: float
    mean_height: float
    min_width: int
    max_width: int
    min_height: int
    max_height: int
    mean_brightness: float
    mean_contrast: float
    mean_entropy: float
    mean_dynamic_range: float
    mean_laplacian_var: float
    mean_colorfulness: float
    mean_fg_fraction: float
    mean_bbox_aspect: float
    mean_bbox_cx: float
    mean_bbox_cy: float


def _gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float32)
    r, g, b = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def _entropy(gray: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(gray.clip(0, 255).astype(np.uint8), bins=bins, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum()) if hist.size else 0.0


def _laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    # Discrete Laplacian
    lap = (
        -4.0 * g
        + np.roll(g, 1, 0)
        + np.roll(g, -1, 0)
        + np.roll(g, 1, 1)
        + np.roll(g, -1, 1)
    )
    return float(lap.var())


def _colorfulness(arr: np.ndarray) -> float:
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return 0.0
    r, g, b = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def compute_image_metrics(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    bbox: Sequence[float] | None = None,
) -> dict[str, float]:
    gray = _gray(image)
    metrics = {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "entropy": _entropy(gray),
        "dynamic_range": float(gray.max() - gray.min()),
        "laplacian_var": _laplacian_var(gray),
        "colorfulness": _colorfulness(image),
        "fg_fraction": foreground_fraction(mask) if mask is not None else 0.0,
        "bbox_aspect": 1.0,
        "bbox_cx": 0.5,
        "bbox_cy": 0.5,
    }
    h, w = gray.shape[:2]
    if bbox is not None and len(bbox) == 4:
        x0, y0, x1, y1 = bbox
        bw, bh = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
        metrics["bbox_aspect"] = float(bw / bh)
        metrics["bbox_cx"] = float(((x0 + x1) * 0.5) / max(w, 1))
        metrics["bbox_cy"] = float(((y0 + y1) * 0.5) / max(h, 1))
    return metrics


def analyze_records(
    records: Sequence[SampleRecord],
    root: str | Path,
    *,
    max_per_class: int | None = 500,
) -> list[ClassStats]:
    """Compute per-class imaging statistics used in the dataset description."""
    root = Path(root)
    by_class: dict[str, list[SampleRecord]] = defaultdict(list)
    for r in records:
        by_class[r.class_name].append(r)

    stats: list[ClassStats] = []
    for class_name, items in sorted(by_class.items()):
        if max_per_class is not None:
            items = items[:max_per_class]
        widths, heights = [], []
        buckets: dict[str, list[float]] = defaultdict(list)
        for rec in items:
            img_path = root / rec.image_path
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                arr = np.array(im.convert("RGB"))
            mask_arr = None
            if rec.mask_path:
                mp = root / rec.mask_path
                if mp.exists():
                    with Image.open(mp) as mm:
                        mask_arr = np.array(mm)
            m = compute_image_metrics(arr, mask_arr, rec.bbox)
            h, w = arr.shape[:2]
            widths.append(w)
            heights.append(h)
            for k, v in m.items():
                buckets[k].append(v)

        def mean(key: str) -> float:
            vals = buckets.get(key, [])
            return float(np.mean(vals)) if vals else 0.0

        stats.append(
            ClassStats(
                class_name=class_name,
                count=len(by_class[class_name]),
                mean_width=float(np.mean(widths)) if widths else 0.0,
                mean_height=float(np.mean(heights)) if heights else 0.0,
                min_width=int(min(widths)) if widths else 0,
                max_width=int(max(widths)) if widths else 0,
                min_height=int(min(heights)) if heights else 0,
                max_height=int(max(heights)) if heights else 0,
                mean_brightness=mean("brightness"),
                mean_contrast=mean("contrast"),
                mean_entropy=mean("entropy"),
                mean_dynamic_range=mean("dynamic_range"),
                mean_laplacian_var=mean("laplacian_var"),
                mean_colorfulness=mean("colorfulness"),
                mean_fg_fraction=mean("fg_fraction"),
                mean_bbox_aspect=mean("bbox_aspect"),
                mean_bbox_cx=mean("bbox_cx"),
                mean_bbox_cy=mean("bbox_cy"),
            )
        )
    return stats


def summarize_dataset(
    root: str | Path,
    *,
    manifest: str | Path | None = None,
    out_json: str | Path | None = None,
    max_per_class: int | None = 500,
) -> dict[str, Any]:
    root = Path(root)
    if manifest is not None:
        records = load_manifest(manifest)
    else:
        records = scan_zenodo_dataset(root, compute_bbox=True, compute_size=True)
    class_counts = Counter(r.class_name for r in records)
    split_counts = Counter(r.split for r in records)
    stats = analyze_records(records, root, max_per_class=max_per_class)
    report = {
        "root": str(root),
        "num_samples": len(records),
        "class_counts": dict(class_counts),
        "split_counts": dict(split_counts),
        "per_class": [asdict(s) for s in stats],
    }
    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote dataset summary to %s", out_json)
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compute microscopy dataset statistics")
    p.add_argument("--root", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="runs/dataset_stats.json")
    p.add_argument("--max-per-class", type=int, default=500)
    args = p.parse_args(argv)
    report = summarize_dataset(args.root, manifest=args.manifest, out_json=args.out, max_per_class=args.max_per_class)
    print(json.dumps({"num_samples": report["num_samples"], "class_counts": report["class_counts"]}, indent=2))


if __name__ == "__main__":
    main()
