"""CLI for segmentation evaluation."""

from __future__ import annotations

import argparse
import json
import logging

from maa.evaluation.segmentation.evaluator import SegmentationEvalConfig, SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate MAA segmentation on test split")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-json", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = SegmentationEvalConfig.from_yaml(args.config) if args.config else SegmentationEvalConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.device is not None:
        cfg.device = args.device

    metrics = SegmentationEvaluator(cfg).run()
    logger.info("Segmentation metrics: %s", json.dumps(metrics, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
