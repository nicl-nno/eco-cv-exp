"""Fixed 70/15/15 train/val/test split helpers.

Splits are deterministic given a seed and sample identifiers so that
classification, detection, and segmentation tasks share the same partition
when fed the same ordered ID list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeVar

from maa.constants import EXPERIMENT_SEEDS

T = TypeVar("T")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


@dataclass(frozen=True)
class SplitIndices:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def _validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")


def compute_split_sizes(
    n: int,
    *,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) with remainder assigned to train."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if n <= 0:
        return 0, 0, 0
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = max(0, n - n_train)
    return n_train, n_val, n_test


def split_indices(
    n: int,
    *,
    seed: int = EXPERIMENT_SEEDS[0],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> SplitIndices:
    """Deterministic index split for integer positions 0..n-1."""
    if n <= 0:
        return SplitIndices((), (), ())

    rng_seed = int(hashlib.sha256(f"{seed}:{n}".encode()).hexdigest(), 16) % (2**32)
    import random

    indices = list(range(n))
    random.Random(rng_seed).shuffle(indices)

    n_train, n_val, n_test = compute_split_sizes(
        n, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    train = tuple(indices[:n_train])
    val = tuple(indices[n_train : n_train + n_val])
    test = tuple(indices[n_train + n_val : n_train + n_val + n_test])
    return SplitIndices(train=train, val=val, test=test)


def split_by_ids(
    ids: Sequence[T],
    *,
    seed: int = EXPERIMENT_SEEDS[0],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[str, tuple[T, ...]]:
    """Split an ordered ID list into train/val/test buckets."""
    idx = split_indices(
        len(ids),
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    return {
        "train": tuple(ids[i] for i in idx.train),
        "val": tuple(ids[i] for i in idx.val),
        "test": tuple(ids[i] for i in idx.test),
    }


def stratified_split_indices(
    labels: Sequence[int],
    *,
    seed: int = EXPERIMENT_SEEDS[0],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> SplitIndices:
    """Per-class 70/15/15 split to preserve label distribution (classification)."""
    import random

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    by_class: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for cls, idxs in sorted(by_class.items()):
        rng = random.Random(int(hashlib.sha256(f"{seed}:{cls}".encode()).hexdigest(), 16) % (2**32))
        rng.shuffle(idxs)
        n_train, n_val, n_test = compute_split_sizes(
            len(idxs), train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
        )
        train.extend(idxs[:n_train])
        val.extend(idxs[n_train : n_train + n_val])
        test.extend(idxs[n_train + n_val : n_train + n_val + n_test])

    rng = random.Random(seed)
    for bucket in (train, val, test):
        rng.shuffle(bucket)

    return SplitIndices(train=tuple(train), val=tuple(val), test=tuple(test))


def save_split_manifest(path: str | Path, split: Mapping[str, Sequence[str]]) -> None:
    """Persist split ID lists as JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: list(v) for k, v in split.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: list(v) for k, v in data.items()}


def filter_items_by_split(
    items: Sequence[T],
    split_name: str,
    *,
    seed: int = EXPERIMENT_SEEDS[0],
    stratified_labels: Sequence[int] | None = None,
) -> list[T]:
    """Return items belonging to train, val, or test."""
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"split_name must be train/val/test, got {split_name!r}")

    if stratified_labels is not None:
        if len(stratified_labels) != len(items):
            raise ValueError("stratified_labels length must match items")
        idx = stratified_split_indices(stratified_labels, seed=seed)
    else:
        idx = split_indices(len(items), seed=seed)

    selected = {"train": idx.train, "val": idx.val, "test": idx.test}[split_name]
    return [items[i] for i in selected]


def multi_seed_splits(
    n: int,
    seeds: Iterable[int] = EXPERIMENT_SEEDS,
) -> dict[int, SplitIndices]:
    """Build splits for all seeds (42, 123, 321)."""
    return {seed: split_indices(n, seed=seed) for seed in seeds}


def assign_fixed_splits(records: Sequence, *, seed: int = EXPERIMENT_SEEDS[0]) -> list:
    """
    Mutate/return sample records with ``.split`` set via stratified 70/15/15.

    Works with objects that have ``class_id`` and mutable ``split`` attributes
    (e.g. ``SampleRecord`` from ``maa.data.zenodo``).
    """
    labels = [int(getattr(r, "class_id", 0)) for r in records]
    idx = stratified_split_indices(labels, seed=seed)
    for i in idx.train:
        records[i].split = "train"
    for i in idx.val:
        records[i].split = "val"
    for i in idx.test:
        records[i].split = "test"
    return list(records)
