"""CLI entrypoint for classification training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from maa.constants import EXPERIMENT_SEEDS
from maa.training.classification.trainer import ClassificationTrainConfig, ClassificationTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MAA classification model")
    p.add_argument("--config", type=str, default=None, help="YAML config (e.g. configs/classification.yaml)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--attention", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--all-seeds", action="store_true", help="Run seeds 42, 123, 321 sequentially")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.config:
        cfg = ClassificationTrainConfig.from_yaml(args.config)
    else:
        cfg = ClassificationTrainConfig()

    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.manifest is not None:
        cfg.manifest = args.manifest
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
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.no_amp:
        cfg.amp = False

    seeds = list(EXPERIMENT_SEEDS) if args.all_seeds else [cfg.seed]
    results = []
    for seed in seeds:
        run_cfg = ClassificationTrainConfig(**{**cfg.__dict__, "seed": seed})
        logger.info("Starting classification training seed=%d", seed)
        trainer = ClassificationTrainer(run_cfg)
        out = trainer.train()
        results.append({"seed": seed, **out})
        logger.info("Finished seed=%d best_val_acc=%s", seed, out.get("best_val_acc"))

    if len(results) > 1:
        accs = [r["best_val_acc"] for r in results if r.get("best_val_acc") is not None]
        if accs:
            logger.info("Mean best val acc over seeds: %.4f", sum(accs) / len(accs))


if __name__ == "__main__":
    main()
