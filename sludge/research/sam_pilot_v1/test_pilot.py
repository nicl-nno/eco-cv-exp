"""Synthetic checks before freezing the inference-only pilot."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from research.sam_pilot_v1.pilot import filter_records, mask_records, measure, select_samples


def test_selection_deterministic_distinct_groups():
    samples = [SimpleNamespace(image_id=f"image{i}", labels=(i % 7 + 1,)) for i in range(30)]
    manifest = pd.DataFrame(
        {"image_id": [s.image_id for s in samples], "group": [f"group{i // 2}" for i in range(30)]}
    )
    selected = select_samples(samples, manifest)
    reversed_selected = select_samples(list(reversed(samples)), manifest)
    assert [s.image_id for s in selected] == [s.image_id for s in reversed_selected]
    groups = manifest.set_index("image_id")["group"].to_dict()
    assert len({groups[s.image_id] for s in selected}) == 12
    assert {label for s in selected for label in s.labels} == set(range(1, 8))


def test_nonfinite_model_output_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        mask_records(torch.full((1, 8, 8), torch.nan), torch.ones(1))


def test_mask_bbox_exclusive_and_empty():
    masks = torch.full((2, 8, 12), -2.0)
    masks[0, 2:6, 3:9] = 2
    result = mask_records(masks, torch.tensor([0.9, 0.2]))
    assert result[0]["box"] == [3, 2, 9, 6]
    assert result[0]["area"] == 24 and result[0]["stability"] == 1
    assert result[1]["box"] == [0, 0, 0, 0] and result[1]["area"] == 0


def test_filter_no_gt_and_nms():
    rows = [
        {"box": [0, 0, 8, 8], "area": 64, "score": 0.99, "stability": 1},
        {"box": [0, 0, 8, 8], "area": 64, "score": 0.98, "stability": 1},
        {"box": [20, 20, 28, 28], "area": 64, "score": 0.5, "stability": 1},
    ]
    assert len(filter_records(rows, True)) == 1
    assert len(filter_records(rows, False)) == 2


def test_one_to_one_matching_duplicates_and_empty():
    rows = [{"box": [0, 0, 8, 8], "score": 0.9}, {"box": [0, 0, 8, 8], "score": 0.8}]
    result = measure(rows, [[0, 0, 8, 8]], [2])
    assert result["tp"] == 1 and result["fp"] == 1 and result["duplicates"] == 1
    assert result["matched_by_class"] == {2: 1}
    assert measure([], [[0, 0, 8, 8]], [1])["fn"] == 1
    assert measure(rows, [], [])["fp"] == 2


def test_iou_uses_exclusive_coordinates():
    result = measure([{"box": [0, 0, 1, 1], "score": 1}], [[1, 0, 2, 1]], [1])
    assert result["tp"] == 0


def test_no_nan_quality_survives():
    rows = [{"box": [0, 0, 8, 8], "area": 64, "score": np.nan, "stability": 1}]
    assert not filter_records(rows, False)
