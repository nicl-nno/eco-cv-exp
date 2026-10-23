"""Freeze three development-only folds before the night follow-up is trained.

CPU metadata preparation only; no model, inference or test-image access.
See GROUPED_CV_2026_10_21.md for the selection rule and six-run budget.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sludge_micro.archives import sha256_file
from sludge_micro.config import ProjectConfig, load_config
from sludge_micro.duplicates import near_pairs
from sludge_micro.group_split import (
    assign_groups,
    group_vectors,
    split_statistics,
    train_filter_effect,
)
from sludge_micro.prepare import load_prepared
from sludge_micro.reporting import read_json, write_json, write_parquet
from sludge_micro.split import load_split, lock_path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/overnight_2026_10_21.yaml"
DESTINATION = ROOT / "experiments_updated_2026_10_02/grouped_cv_2026_10_21"
EXPECTED_SOURCE_SHA = "6f88c2d3404b9028ea0fa38b81ac4ebaab19d6881d83a9c426e3e2c2ff09675c"
FOLD_NAMES = {"train": 0, "validation": 1, "test": 2}


def development_only(original: pd.DataFrame) -> pd.DataFrame:
    """Exclude original test membership and reject pre-existing group/hash leaks."""
    test = original[original["split"] == "test"]
    development = original[original["split"].isin(["train", "validation"])].copy()
    if development.empty or test.empty:
        raise ValueError("Expected an existing train/validation/test split")
    for key in ("image_id", "group", "file_sha256", "pixel_sha256"):
        if set(development[key]) & set(test[key]):
            raise ValueError(f"Original test overlaps development: {key}")
    if development["image_id"].duplicated().any():
        raise ValueError("Duplicate development image IDs")
    return development


def check_fold(manifest: pd.DataFrame, original: pd.DataFrame, hamming: int) -> None:
    """Two-way equivalent of the package's three-way independence check."""
    development = development_only(original)
    if set(manifest["image_id"]) != set(development["image_id"]):
        raise ValueError("Fold membership differs from development-only pool")
    if len(manifest) != len(development) or set(manifest["split"]) != {"train", "validation"}:
        raise ValueError("Expected two nonempty, nonduplicated development subsets")
    for key in ("group", "pixel_sha256", "file_sha256"):
        if (manifest.groupby(key)["split"].nunique() > 1).any():
            raise ValueError(f"Fold leakage: {key}")
    labels = manifest["split"].tolist()
    if any(labels[i] != labels[j] for i, j, _ in near_pairs(manifest["dhash"].tolist(), hamming)):
        raise ValueError("Fold leakage: near-duplicate dHash")
    # The original group for every frame must be retained, not relabelled to hide leakage.
    groups = original.set_index("image_id")["group"]
    if not (manifest["group"].to_numpy() == manifest["image_id"].map(groups).to_numpy()).all():
        raise ValueError("Original group identity changed")


def allocation_config(config: ProjectConfig) -> ProjectConfig:
    return config.model_copy(
        update={
            "runtime": config.runtime.model_copy(update={"seed": 20261021}),
            "split": config.split.model_copy(
                update={
                    "target_fractions": {name: 1 / 3 for name in FOLD_NAMES},
                    "grouping": config.split.grouping.model_copy(update={"search_iterations": 200}),
                }
            ),
        }
    )


def prepare() -> dict:
    config = load_config(BASE)
    original_lock = read_json(lock_path(config))
    if original_lock["manifest_sha256"] != EXPECTED_SOURCE_SHA:
        raise ValueError("Original locked split changed")
    original = load_split(config)
    development = development_only(original).reset_index(drop=True)
    _, all_annotations = load_prepared(config)
    # Do not use test annotations, including in fold balancing or support statistics.
    annotations = all_annotations[all_annotations["image_id"].isin(development["image_id"])].copy()
    ignored_ids = set(annotations.loc[annotations["supervision"] == "ignore", "image_id"])
    vectors = group_vectors(development, annotations, config.classes.names)
    usable = annotations[~annotations["image_id"].isin(ignored_ids)]
    training_vectors = group_vectors(development, usable, config.classes.names)
    vectors = vectors.join(training_vectors.drop(columns="images").add_prefix("usable:"))
    registered = DESTINATION / "fold_registration.json"
    if registered.exists():
        registration = read_json(registered)
        if registration["source_manifest_sha256"] != EXPECTED_SOURCE_SHA:
            raise ValueError("Existing fold registration has a different source")
        for fold in registration["folds"]:
            path = DESTINATION / fold["manifest"]
            if sha256_file(path) != fold["manifest_sha256"]:
                raise ValueError("Registered fold was modified")
            check_fold(pd.read_parquet(path), original, config.audit.near_duplicate_max_hamming)
        return registration
    assignment, cost = assign_groups(allocation_config(config), vectors)
    development["cv_fold"] = development["group"].map(assignment).map(FOLD_NAMES)
    if development["cv_fold"].isna().any():
        raise ValueError("Unassigned development group")
    folds = []
    for fold in range(3):
        manifest = development.copy()
        manifest["split"] = "train"
        manifest.loc[manifest["cv_fold"] == fold, "split"] = "validation"
        check_fold(manifest, original, config.audit.near_duplicate_max_hamming)
        folder = DESTINATION / f"fold_{fold}"
        path = folder / "split_manifest.parquet"
        if path.exists() or (folder / "split_lock.json").exists():
            raise ValueError("Incomplete prior preparation found; inspect it before retrying")
        write_parquet(path, manifest)
        stats = split_statistics(config, manifest, annotations)
        effect = train_filter_effect(config, manifest, annotations)
        lock = {
            "protocol": {
                "kind": "grouped_development_cv",
                "fold": fold,
                "folds": 3,
                "seed": 20261021,
                "restarts": 200,
                "original_test_excluded": True,
            },
            "source_manifest_sha256": EXPECTED_SOURCE_SHA,
            "data_variant": config.prepare.variant,
            "manifest": path.name,
            "manifest_sha256": sha256_file(path),
            "groups": int(manifest["group"].nunique()),
            "splits": stats,
            "train_filter": effect,
            "missing_training_classes": [
                c for c, count in effect["objects_after"].items() if not count
            ],
        }
        write_json(folder / "split_lock.json", lock)
        # Refuse accidental CLI test evaluation before prediction; no test rows exist.
        write_json(
            folder / "test_usage.json", {"disabled": True, "reason": "Development CV: no test"}
        )
        folds.append(
            {
                "fold": fold,
                "manifest": path.relative_to(DESTINATION).as_posix(),
                "manifest_sha256": lock["manifest_sha256"],
                "splits": stats,
                "train_filter": effect,
                "missing_training_classes": lock["missing_training_classes"],
            }
        )
    result = {
        "source_manifest_sha256": EXPECTED_SOURCE_SHA,
        "development_images": len(development),
        "development_groups": int(development["group"].nunique()),
        "assignment_cost": cost,
        "selection": "Best non-control three-seed mean, only if greater than control",
        "seed": 20261021,
        "fixed_confidence": 0.2,
        "max_training_runs": 6,
        "folds": folds,
        "independent_external_test": False,
    }
    write_json(registered, result)
    return result


if __name__ == "__main__":
    import json

    result = prepare()
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=2))
