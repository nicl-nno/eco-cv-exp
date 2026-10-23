# Baseline configs

Per-model launch YAMLs for every backbone × attention pair.

| Directory | Count | Pattern |
|-----------|------:|---------|
| [`classification/`](classification/) | 30 | `{backbone}_{attention}.yaml` |
| [`detection/`](detection/) | 30 | `{backbone}_{attention}.yaml` |
| [`segmentation/`](segmentation/) | 25 | `{backbone}_{attention}.yaml` |

Attentions: `none`, `se`, `sa`, `cbam`, `maa`.

Shared protocol defaults live in [`default.yaml`](default.yaml). Leave-one-out MAA ablations use [`ablation.yaml`](ablation.yaml).
