"""Development-fold isolation without training or opening microscopy images."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def cv_module():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_grouped_cv.py"
    spec = importlib.util.spec_from_file_location("grouped_cv_preparation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def original():
    return pd.DataFrame(
        [
            {
                "image_id": f"img{i}",
                "group": f"g{i // 2}",
                "file_sha256": f"file{i}",
                "pixel_sha256": f"pixel{i}",
                "dhash": f"{i:016x}",
                "split": "test" if i >= 4 else ("train" if i < 2 else "validation"),
            }
            for i in range(6)
        ]
    )


def test_development_pool_excludes_all_test_membership(cv_module, original):
    development = cv_module.development_only(original)
    assert set(development.image_id) == {"img0", "img1", "img2", "img3"}
    cv_module.check_fold(development, original, 0)


@pytest.mark.parametrize("key", ["group", "pixel_sha256", "file_sha256"])
def test_original_test_leak_is_rejected(cv_module, original, key):
    original.loc[4, key] = original.loc[0, key]
    with pytest.raises(ValueError, match="Original test overlaps"):
        cv_module.development_only(original)


def test_group_cannot_be_split_across_train_and_validation(cv_module, original):
    development = cv_module.development_only(original)
    development.loc[0, "split"] = "validation"
    with pytest.raises(ValueError, match="Fold leakage: group"):
        cv_module.check_fold(development, original, 0)


def test_test_frame_cannot_be_added_to_a_fold(cv_module, original):
    manifest = original.copy()
    manifest.loc[manifest["split"] == "test", "split"] = "train"
    with pytest.raises(ValueError, match="membership differs"):
        cv_module.check_fold(manifest, original, 0)


def test_near_duplicate_crossing_is_rejected(cv_module, original):
    development = cv_module.development_only(original)
    development.loc[2, "dhash"] = development.loc[0, "dhash"]
    with pytest.raises(ValueError, match="near-duplicate"):
        cv_module.check_fold(development, original, 0)


def test_relabelled_original_group_is_rejected(cv_module, original):
    development = cv_module.development_only(original)
    development.loc[0, "group"] = "new_group"
    with pytest.raises(ValueError, match="group identity changed"):
        cv_module.check_fold(development, original, 0)
