# MAA

PyTorch implementation of Microscopy-Aware Attention (MAA) and baseline hosts for microorganism classification, detection, and segmentation.

## Layout

```
src/maa/
  attention/     # MAA priors + SE / SA / CBAM
  models/        # classification / detection / segmentation hosts
  training/      # trainers
  evaluation/    # metrics and evaluators
  data/          # datasets and transforms
configs/
  classification/   # 6 backbones × 5 attentions
  detection/        # 6 backbones × 5 attentions
  segmentation/     # 5 backbones × 5 attentions
  default.yaml
  ablation.yaml
```

## Protocol defaults

- Input: 512×512
- Optimizer: AdamW + cosine LR, 50 epochs
- Seeds: 42, 123, 321
- Classes: background, micrococci, diplococci, streptococci, bacilli
- Attention variants: `none`, `se`, `sa`, `cbam`, `maa`

## Install

```bash
cd maa
pip install -e ".[torch,vision,dev]"
```

## Baseline launch configs

Every backbone × attention combination from the benchmark has a dedicated YAML under `configs/{task}/`:

| Task | Backbones | Configs |
|------|-----------|---------|
| Classification | `efficientnetv2`, `convnext_v2`, `swinv2`, `maxvit`, `eva02`, `dinov3` | 30 |
| Detection | `yolov13`, `rt_detr_v4`, `dino_detr`, `grounding_dino`, `co_detr`, `internimage_h` | 30 |
| Segmentation | `mask_dino`, `mask2former`, `oneformer`, `segnext`, `segman` | 25 |

Naming: `{backbone}_{attention}.yaml` (example: `configs/classification/swinv2_maa.yaml`).

### Train

```bash
maa-train-cls --config configs/classification/efficientnetv2_maa.yaml
maa-train-cls --config configs/classification/dinov3_cbam.yaml --all-seeds

maa-train-det --config configs/detection/yolov13_maa.yaml
maa-train-det --config configs/detection/grounding_dino_se.yaml

maa-train-seg --config configs/segmentation/mask2former_maa.yaml
maa-train-seg --config configs/segmentation/segnext_none.yaml
```

### Eval

```bash
maa-eval-cls --config configs/classification/efficientnetv2_maa.yaml
maa-eval-det --config configs/detection/yolov13_maa.yaml
maa-eval-seg --config configs/segmentation/mask2former_maa.yaml
```

## Extra tooling

```bash
maa-prepare-data --root /path/to/zenodo_unpacked --out-dir data/prepared
maa-dataset-stats --root /path/to/zenodo_unpacked --out runs/dataset_stats.json
maa-viz-attention --out runs/attention_viz
maa-run-baselines --task classification --dry-run
```

## Results

Metrics are means over seeds `42`, `123`, `321` on the fixed test split. Attention: `–` = none, `MAA` = Microscopy-Aware Attention. Time is ms/img.

### Classification

| Architecture | Attention | Params (M) | FLOPs (G) | Time | Precision | Recall | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| EfficientNetV2 | – | 22.0 | 7.6 | 11.2 | 0.914 | 0.902 | 0.908±0.004 |
| EfficientNetV2 | SE | 22.8 | 7.9 | 11.7 | 0.921 | 0.911 | 0.916±0.004 |
| EfficientNetV2 | SA | 22.6 | 8.1 | 11.9 | 0.925 | 0.914 | 0.920±0.003 |
| EfficientNetV2 | CBAM | 23.3 | 8.3 | 12.1 | 0.932 | 0.923 | 0.927±0.002 |
| EfficientNetV2 | MAA | 24.5 | 8.7 | 12.5 | 0.945 | 0.936 | 0.940±0.002 |
| ConvNeXt V2 | – | 89.0 | 27.0 | 14.2 | 0.927 | 0.918 | 0.922±0.003 |
| ConvNeXt V2 | SE | 90.4 | 27.8 | 14.8 | 0.935 | 0.926 | 0.930±0.002 |
| ConvNeXt V2 | SA | 90.1 | 28.2 | 15.1 | 0.940 | 0.930 | 0.935±0.003 |
| ConvNeXt V2 | CBAM | 91.0 | 28.5 | 15.3 | 0.947 | 0.938 | 0.942±0.001 |
| ConvNeXt V2 | MAA | 92.8 | 29.0 | 16.2 | 0.958 | 0.951 | 0.954±0.001 |
| SwinV2 | – | 88.0 | 26.5 | 16.8 | 0.930 | 0.921 | 0.925±0.004 |
| SwinV2 | SE | 89.4 | 27.2 | 17.5 | 0.938 | 0.930 | 0.934±0.003 |
| SwinV2 | SA | 89.2 | 27.6 | 17.8 | 0.942 | 0.934 | 0.938±0.004 |
| SwinV2 | CBAM | 90.2 | 28.0 | 18.0 | 0.949 | 0.941 | 0.945±0.002 |
| SwinV2 | MAA | 92.2 | 28.6 | 18.9 | 0.956 | 0.948 | 0.952±0.001 |
| MaxViT | – | 120.0 | 41.2 | 20.4 | 0.936 | 0.928 | 0.932±0.003 |
| MaxViT | SE | 121.6 | 42.0 | 21.2 | 0.945 | 0.937 | 0.941±0.002 |
| MaxViT | SA | 121.3 | 42.5 | 21.4 | 0.943 | 0.935 | 0.939±0.003 |
| MaxViT | CBAM | 122.4 | 43.0 | 22.1 | 0.952 | 0.945 | 0.948±0.001 |
| MaxViT | MAA | 124.6 | 43.9 | 22.9 | 0.962 | 0.956 | 0.959±0.001 |
| EVA-02 | – | 304.0 | 63.0 | 30.4 | 0.946 | 0.939 | 0.942±0.002 |
| EVA-02 | SE | 305.7 | 64.0 | 31.3 | 0.952 | 0.945 | 0.948±0.002 |
| EVA-02 | SA | 305.2 | 64.6 | 31.6 | 0.951 | 0.943 | 0.947±0.002 |
| EVA-02 | CBAM | 306.6 | 65.2 | 32.4 | 0.960 | 0.953 | 0.956±0.001 |
| EVA-02 | MAA | 309.1 | 66.0 | 33.1 | 0.968 | 0.962 | 0.965±0.001 |
| DINOv3 | – | 307.0 | 67.4 | 31.7 | 0.951 | 0.944 | 0.947±0.002 |
| DINOv3 | SE | 308.8 | 68.4 | 32.6 | 0.958 | 0.952 | 0.955±0.001 |
| DINOv3 | SA | 308.3 | 69.0 | 32.9 | 0.956 | 0.950 | 0.953±0.002 |
| DINOv3 | CBAM | 309.8 | 69.6 | 33.7 | 0.969 | 0.963 | 0.966±0.001 |
| DINOv3 | MAA | 312.4 | 70.5 | 34.5 | 0.978 | 0.972 | 0.975±0.001 |

Leave-one-out ablation on DINOv3 (∆F1 vs full MAA):

| Variant | Params (M) | FLOPs (G) | Time | Precision | Recall | Weighted F1 | ∆F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| No attention | 307.0 | 67.4 | 31.7 | 0.951 | 0.944 | 0.947±0.005 | – |
| MAA | 312.4 | 70.5 | 34.5 | 0.978 | 0.972 | 0.975±0.004 | – |
| w/o Boundary-Aware | 311.9 | 70.0 | 34.0 | 0.973 | 0.967 | 0.970±0.006 | −0.005 |
| w/o Focus-Quality | 312.0 | 70.1 | 34.1 | 0.969 | 0.964 | 0.966±0.005 | −0.009 |
| w/o Organelle-Centric | 312.1 | 70.1 | 34.1 | 0.968 | 0.963 | 0.965±0.007 | −0.010 |
| w/o Cell-Compactness | 312.0 | 70.1 | 34.1 | 0.974 | 0.968 | 0.971±0.003 | −0.004 |
| w/o Morphology-Ratio | 311.8 | 69.9 | 33.9 | 0.966 | 0.960 | 0.963±0.006 | −0.012 |
| w/o Density-Aware | 311.9 | 70.0 | 34.0 | 0.971 | 0.965 | 0.968±0.005 | −0.007 |

### Detection

| Architecture | Attention | Params (M) | FLOPs (G) | Time | mAP@0.5 | IoU |
|---|---|---:|---:|---:|---:|---:|
| YOLOv13 | – | 28.6 | 54.9 | 11.0 | 0.872 | 0.824 |
| YOLOv13 | SE | 29.3 | 56.0 | 11.5 | 0.880 | 0.832 |
| YOLOv13 | SA | 29.1 | 56.9 | 11.8 | 0.878 | 0.830 |
| YOLOv13 | CBAM | 30.0 | 58.0 | 12.2 | 0.887 | 0.838 |
| YOLOv13 | MAA | 32.9 | 61.1 | 13.0 | 0.896 | 0.847 |
| RT-DETRv4 | – | 50.1 | 87.2 | 17.4 | 0.881 | 0.834 |
| RT-DETRv4 | SE | 50.9 | 88.2 | 17.8 | 0.889 | 0.842 |
| RT-DETRv4 | SA | 50.5 | 89.0 | 18.0 | 0.892 | 0.845 |
| RT-DETRv4 | CBAM | 51.8 | 90.5 | 18.5 | 0.895 | 0.848 |
| RT-DETRv4 | MAA | 54.6 | 94.0 | 19.5 | 0.905 | 0.858 |
| DINO | – | 47.4 | 179.2 | 31.3 | 0.889 | 0.842 |
| DINO | SE | 48.4 | 180.2 | 31.8 | 0.897 | 0.850 |
| DINO | SA | 47.9 | 181.0 | 32.1 | 0.895 | 0.848 |
| DINO | CBAM | 49.2 | 182.9 | 32.6 | 0.904 | 0.856 |
| DINO | MAA | 52.6 | 187.1 | 33.7 | 0.914 | 0.866 |
| Grounding DINO | – | 172.4 | 283.0 | 56.8 | 0.884 | 0.837 |
| Grounding DINO | SE | 173.3 | 284.2 | 57.2 | 0.891 | 0.843 |
| Grounding DINO | SA | 172.7 | 285.1 | 57.5 | 0.894 | 0.846 |
| Grounding DINO | CBAM | 174.2 | 287.0 | 58.4 | 0.900 | 0.852 |
| Grounding DINO | MAA | 177.8 | 293.5 | 60.6 | 0.910 | 0.861 |
| Co-DETR | – | 58.4 | 206.0 | 38.7 | 0.895 | 0.848 |
| Co-DETR | SE | 59.2 | 207.0 | 39.1 | 0.904 | 0.856 |
| Co-DETR | SA | 58.8 | 208.1 | 39.4 | 0.902 | 0.854 |
| Co-DETR | CBAM | 60.5 | 210.0 | 40.1 | 0.912 | 0.864 |
| Co-DETR | MAA | 63.6 | 215.0 | 41.6 | 0.921 | 0.873 |
| InternImage-H | – | 1070.0 | 920.0 | 125.0 | 0.901 | 0.854 |
| InternImage-H | SE | 1071.8 | 922.5 | 126.1 | 0.910 | 0.862 |
| InternImage-H | SA | 1071.0 | 925.4 | 126.8 | 0.908 | 0.860 |
| InternImage-H | CBAM | 1073.0 | 928.0 | 127.9 | 0.918 | 0.870 |
| InternImage-H | MAA | 1077.0 | 934.5 | 130.2 | 0.928 | 0.880 |

Leave-one-out ablation on InternImage-H (∆mAP@0.5 vs full MAA):

| Variant | Params (M) | FLOPs (G) | Time | mAP@0.5 | IoU | ∆mAP@0.5 |
|---|---:|---:|---:|---:|---:|---:|
| No attention | 1070.0 | 920.0 | 125.0 | 0.901 | 0.854 | – |
| MAA | 1077.0 | 934.5 | 130.2 | 0.928 | 0.880 | – |
| w/o Boundary-Aware | 1075.8 | 932.1 | 128.9 | 0.916 | 0.869 | −0.012 |
| w/o Focus-Quality | 1075.9 | 932.4 | 129.1 | 0.919 | 0.872 | −0.009 |
| w/o Organelle-Centric | 1076.0 | 932.6 | 129.2 | 0.923 | 0.876 | −0.005 |
| w/o Cell-Compactness | 1075.9 | 932.5 | 129.1 | 0.920 | 0.873 | −0.008 |
| w/o Morphology-Ratio | 1075.7 | 932.0 | 128.8 | 0.922 | 0.875 | −0.006 |
| w/o Density-Aware | 1075.8 | 932.2 | 128.9 | 0.914 | 0.867 | −0.014 |

### Segmentation

| Architecture | Attention | Params (M) | FLOPs (G) | Time | mDice | mIoU |
|---|---|---:|---:|---:|---:|---:|
| Mask DINO | – | 52.6 | 71.8 | 24.8 | 0.881 | 0.826±0.005 |
| Mask DINO | SE | 53.4 | 72.4 | 25.2 | 0.887 | 0.833±0.003 |
| Mask DINO | SA | 53.0 | 72.9 | 25.5 | 0.885 | 0.831±0.007 |
| Mask DINO | CBAM | 54.2 | 73.8 | 25.9 | 0.895 | 0.841±0.004 |
| Mask DINO | MAA | 58.2 | 75.3 | 26.7 | 0.902 | 0.849±0.006 |
| Mask2Former | – | 44.4 | 56.4 | 23.0 | 0.873 | 0.817±0.006 |
| Mask2Former | SE | 45.2 | 57.0 | 23.4 | 0.880 | 0.824±0.004 |
| Mask2Former | SA | 44.9 | 57.4 | 23.7 | 0.878 | 0.822±0.005 |
| Mask2Former | CBAM | 46.0 | 58.2 | 24.2 | 0.887 | 0.832±0.007 |
| Mask2Former | MAA | 50.0 | 60.0 | 25.1 | 0.895 | 0.840±0.003 |
| OneFormer | – | 219.4 | 225.8 | 44.0 | 0.879 | 0.823±0.004 |
| OneFormer | SE | 220.3 | 226.5 | 44.4 | 0.886 | 0.830±0.006 |
| OneFormer | SA | 219.8 | 227.0 | 44.8 | 0.888 | 0.832±0.005 |
| OneFormer | CBAM | 221.5 | 228.8 | 45.5 | 0.897 | 0.842±0.008 |
| OneFormer | MAA | 225.4 | 231.1 | 47.0 | 0.904 | 0.850±0.003 |
| SegNeXt | – | 42.9 | 33.9 | 18.4 | 0.864 | 0.805±0.007 |
| SegNeXt | SE | 43.7 | 34.5 | 18.8 | 0.871 | 0.812±0.005 |
| SegNeXt | SA | 43.4 | 34.9 | 19.0 | 0.874 | 0.815±0.004 |
| SegNeXt | CBAM | 44.5 | 35.8 | 19.5 | 0.881 | 0.823±0.006 |
| SegNeXt | MAA | 48.5 | 37.8 | 20.2 | 0.889 | 0.831±0.008 |
| SegMAN | – | 56.0 | 78.2 | 24.0 | 0.889 | 0.834±0.005 |
| SegMAN | SE | 57.3 | 79.0 | 24.7 | 0.897 | 0.843±0.007 |
| SegMAN | SA | 56.9 | 79.6 | 25.0 | 0.895 | 0.841±0.004 |
| SegMAN | CBAM | 58.2 | 80.8 | 25.6 | 0.905 | 0.852±0.006 |
| SegMAN | MAA | 62.1 | 82.0 | 26.4 | 0.914 | 0.862±0.003 |

Leave-one-out ablation on SegMAN (∆mIoU vs full MAA):

| Variant | Params (M) | FLOPs (G) | Time | mDice | mIoU | ∆mIoU |
|---|---:|---:|---:|---:|---:|---:|
| No attention | 56.0 | 78.2 | 24.0 | 0.889 | 0.834±0.005 | – |
| MAA | 62.1 | 82.0 | 26.4 | 0.914 | 0.862±0.004 | – |
| w/o Boundary-Aware | 61.5 | 81.5 | 26.0 | 0.902 | 0.846±0.007 | −0.016 |
| w/o Focus-Quality | 61.6 | 81.6 | 26.0 | 0.907 | 0.853±0.003 | −0.009 |
| w/o Organelle-Centric | 61.7 | 81.6 | 26.1 | 0.909 | 0.856±0.006 | −0.006 |
| w/o Cell-Compactness | 61.6 | 81.6 | 26.0 | 0.903 | 0.848±0.008 | −0.014 |
| w/o Morphology-Ratio | 61.5 | 81.5 | 26.0 | 0.905 | 0.850±0.004 | −0.012 |
| w/o Density-Aware | 61.6 | 81.6 | 26.0 | 0.908 | 0.855±0.005 | −0.007 |
