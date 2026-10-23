"""Check fold isolation and copy-paste labels, and save a visual example."""
from pathlib import Path
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold

import data


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "task" / "coco_export"
    images, annotations = data.read_coco(root)
    ids = np.array(sorted(images))
    train_idx, val_idx = next(GroupKFold(5).split(ids, groups=ids))
    train_ids, val_ids = ids[train_idx].tolist(), set(ids[val_idx])
    original_augment = data.augment
    data.augment = lambda image, mask: (image, mask)
    try:
        baseline = data.TrainPatches(root, images, annotations, train_ids)
        synthetic = data.TrainPatches(root, images, annotations, train_ids,
                                      copy_paste_probability=1.0, copy_paste_max_objects=2)
        assert all(donor_id not in val_ids for donors in synthetic.donors.values()
                   for donor_id, _ in donors)
        found = False
        for index in range(20):
            random.seed(1000 + index)
            plain_image, plain_mask = baseline[index]
            random.seed(1000 + index)
            made_image, made_mask = synthetic[index]
            assert np.array_equal(plain_mask[plain_mask != 0], made_mask[plain_mask != 0])
            assert not np.any((plain_mask == data.IGNORE) & (made_mask != data.IGNORE))
            added = (plain_mask == 0) & np.isin(made_mask, [1, 2])
            if added.any():
                assert np.any(plain_image[added] != made_image[added])
                if not found:
                    destination = Path(__file__).resolve().parent / "external" / "copy_paste_preview.png"
                    before, after = Image.fromarray(plain_image), Image.fromarray(made_image)
                    canvas = Image.new("RGB", (data.PATCH * 2, data.PATCH))
                    canvas.paste(before, (0, 0))
                    canvas.paste(after, (data.PATCH, 0))
                    canvas.save(destination)
                    print(f"preview={destination} added_pixels={int(added.sum())}")
                found = True
        assert found, "No pasted objects in 20 trials"
        print("copy-paste smoke test passed; validation images never used as donors")
    finally:
        data.augment = original_augment


if __name__ == "__main__":
    main()
