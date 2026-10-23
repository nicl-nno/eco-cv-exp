"""Classification dataset: ImageFolder layout or manifest CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from maa.constants import CLASS_TO_IDX, CLASS_NAMES, NUM_CLASSES
from maa.data.splits import filter_items_by_split, load_split_manifest, stratified_split_indices


@dataclass(frozen=True)
class ClassificationSample:
    path: Path
    label: int
    class_name: str


def _read_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _discover_imagefolder(root: Path) -> list[ClassificationSample]:
    samples: list[ClassificationSample] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        class_name = class_dir.name.lower()
        if class_name not in CLASS_TO_IDX:
            continue
        label = CLASS_TO_IDX[class_name]
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                samples.append(ClassificationSample(path=path, label=label, class_name=class_name))
    if not samples:
        raise FileNotFoundError(f"No labeled images found under {root}")
    return samples


def _discover_manifest_csv(manifest: Path, root: Optional[Path]) -> list[ClassificationSample]:
    samples: list[ClassificationSample] = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV has no header: {manifest}")
        path_key = next((k for k in reader.fieldnames if k.lower() in {"path", "filepath", "image", "file"}), None)
        label_key = next((k for k in reader.fieldnames if k.lower() in {"label", "class", "class_name", "category"}), None)
        if path_key is None or label_key is None:
            raise ValueError("Manifest must contain path and label/class columns")
        for row in reader:
            rel = row[path_key].strip()
            path = Path(rel)
            if not path.is_absolute():
                base = root or manifest.parent
                path = (base / rel).resolve()
            raw_label = row[label_key].strip()
            if raw_label.isdigit():
                label = int(raw_label)
                class_name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label)
            else:
                class_name = raw_label.lower()
                if class_name not in CLASS_TO_IDX:
                    raise ValueError(f"Unknown class {raw_label!r} in {manifest}")
                label = CLASS_TO_IDX[class_name]
            samples.append(ClassificationSample(path=path, label=label, class_name=class_name))
    if not samples:
        raise FileNotFoundError(f"No rows loaded from manifest {manifest}")
    return samples


class MicroscopyClassificationDataset(Dataset):
    """
    Microorganism classification dataset.

    Supports:
    - ImageFolder: ``root/<class_name>/*.png``
    - Manifest CSV with columns ``path`` and ``label`` (or ``class``)

    When ``split`` is set, applies the fixed 70/15/15 protocol (stratified by label).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: str | Path | None = None,
        split: str | None = None,
        split_manifest: str | Path | None = None,
        seed: int = 42,
        transform: Callable[[Any], torch.Tensor] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.seed = seed

        if manifest is not None:
            self.samples = _discover_manifest_csv(Path(manifest), self.root)
        else:
            self.samples = _discover_imagefolder(self.root)

        if split_manifest is not None:
            split_map = load_split_manifest(split_manifest)
            allowed = {Path(p).resolve() for p in split_map[split or "train"]}
            self.samples = [s for s in self.samples if s.path.resolve() in allowed]
        elif split is not None:
            labels = [s.label for s in self.samples]
            idx = stratified_split_indices(labels, seed=seed)
            bucket = {"train": idx.train, "val": idx.val, "test": idx.test}[split]
            self.samples = [self.samples[i] for i in bucket]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = _read_image(sample.path)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "class_name": sample.class_name,
            "path": str(sample.path),
        }

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for s in self.samples:
            counts[s.label] = counts.get(s.label, 0) + 1
        return counts

    def compute_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for long-tail CE (optional)."""
        counts = self.class_counts()
        total = sum(counts.values())
        weights = torch.ones(NUM_CLASSES, dtype=torch.float32)
        for cls in range(NUM_CLASSES):
            c = counts.get(cls, 0)
            if c > 0:
                weights[cls] = total / (NUM_CLASSES * c)
        return weights
