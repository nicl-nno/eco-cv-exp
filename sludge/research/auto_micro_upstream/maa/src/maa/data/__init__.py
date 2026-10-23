"""Data loading utilities for the MAA microscopy benchmark."""

from maa.data.collate import (
    COLLATE_FNS,
    build_collate_fn,
    collate_classification,
    collate_detection,
    collate_segmentation,
)
from maa.data.dataset_classification import MicroscopyClassificationDataset
from maa.data.dataset_detection import COCODetectionDataset
from maa.data.dataset_segmentation import MicroscopySegmentationDataset
from maa.data.splits import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SplitIndices,
    assign_fixed_splits,
    compute_split_sizes,
    filter_items_by_split,
    load_split_manifest,
    multi_seed_splits,
    save_split_manifest,
    split_by_ids,
    split_indices,
    stratified_split_indices,
)
from maa.data.statistics import analyze_records, summarize_dataset
from maa.data.transforms import (
    TransformConfig,
    build_classification_transforms,
    build_detection_transforms,
    build_segmentation_transforms,
)
from maa.data.zenodo import (
    SampleRecord,
    export_coco_detection,
    load_manifest,
    scan_zenodo_dataset,
    write_manifest,
)

__all__ = [
    "COCODetectionDataset",
    "MicroscopyClassificationDataset",
    "MicroscopySegmentationDataset",
    "SampleRecord",
    "TransformConfig",
    "SplitIndices",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "analyze_records",
    "assign_fixed_splits",
    "build_classification_transforms",
    "build_detection_transforms",
    "build_segmentation_transforms",
    "build_collate_fn",
    "collate_classification",
    "collate_detection",
    "collate_segmentation",
    "COLLATE_FNS",
    "compute_split_sizes",
    "export_coco_detection",
    "filter_items_by_split",
    "load_manifest",
    "load_split_manifest",
    "multi_seed_splits",
    "save_split_manifest",
    "scan_zenodo_dataset",
    "split_by_ids",
    "split_indices",
    "stratified_split_indices",
    "summarize_dataset",
    "write_manifest",
]
