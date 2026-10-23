"""CLI to prepare Zenodo dataset manifests / COCO exports / stats."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from maa.constants import EXPERIMENT_SEEDS
from maa.data.splits import assign_fixed_splits
from maa.data.statistics import summarize_dataset
from maa.data.zenodo import export_coco_detection, scan_zenodo_dataset, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare MAA microscopy dataset artefacts")
    p.add_argument("--root", required=True, help="Unpacked Zenodo dataset root")
    p.add_argument("--out-dir", default="data/prepared")
    p.add_argument("--limit-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=EXPERIMENT_SEEDS[0])
    p.add_argument("--skip-stats", action="store_true")
    p.add_argument("--skip-coco", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = scan_zenodo_dataset(
        root,
        compute_bbox=True,
        compute_size=True,
        limit_per_class=args.limit_per_class,
    )
    # Prefer split helper if available; otherwise simple hash split
    try:
        records = assign_fixed_splits(records, seed=args.seed)
    except Exception:
        import hashlib

        for r in records:
            h = int(hashlib.md5(r.image_path.encode()).hexdigest(), 16) % 100
            r.split = "train" if h < 70 else ("val" if h < 85 else "test")

    manifest_path = out / "manifest.json"
    write_manifest(records, manifest_path)

    if not args.skip_coco:
        for split in ("train", "val", "test"):
            export_coco_detection(records, out / f"instances_{split}.json", split=split)

    if not args.skip_stats:
        summarize_dataset(
            root,
            manifest=manifest_path,
            out_json=out / "dataset_stats.json",
            max_per_class=min(args.limit_per_class or 500, 500),
        )

    logger.info("Prepared dataset artefacts in %s", out)


if __name__ == "__main__":
    main()
