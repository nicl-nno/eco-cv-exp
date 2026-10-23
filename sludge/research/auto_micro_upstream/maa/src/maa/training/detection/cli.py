"""CLI entrypoint for detection training."""

from __future__ import annotations

import argparse
import logging

from maa.constants import EXPERIMENT_SEEDS
from maa.training.detection.trainer import DetectionTrainConfig, DetectionTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MAA detection model")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--annotation-file", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--attention", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--all-seeds", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = DetectionTrainConfig.from_yaml(args.config) if args.config else DetectionTrainConfig()

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.annotation_file is not None:
        cfg.annotation_file = args.annotation_file
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.backbone is not None:
        cfg.backbone = args.backbone
    if args.attention is not None:
        cfg.attention = args.attention
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.no_amp:
        cfg.amp = False

    seeds = list(EXPERIMENT_SEEDS) if args.all_seeds else [cfg.seed]
    for seed in seeds:
        run_cfg = DetectionTrainConfig(**{**cfg.__dict__, "seed": seed})
        logger.info("Starting detection training seed=%d", seed)
        DetectionTrainer(run_cfg).train()


if __name__ == "__main__":
    main()
