"""Binary larva segmentation pretraining on a licensed, distinct Zenodo dataset.

The source's 0/1 categories are NOT treated as target-stage labels: both are
mapped to larva, and only the learned ResNet34 encoder is transferred.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from data import augment
from model import ResNet34UNet
from train import atomic_json, atomic_torch_save, seed_everything


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "external" / "zenodo" / "BSF_Larvae_v1"
OUTPUT = HERE / "runs" / "external_pretrain"
PATCH = 512


class ZenodoLarvae(Dataset):
    def __init__(self) -> None:
        paths = sorted((SOURCE / "Images").glob("*.jpg"))
        # Roboflow variants of the same original must not masquerade as new scenes.
        unique = {}
        for path in paths:
            original = path.name.split("_jpg.rf.")[0]
            unique.setdefault(original, path)
        self.paths = list(unique.values())
        if len(self.paths) != 185:
            raise ValueError(f"Expected 185 original stems, got {len(self.paths)}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = np.asarray(Image.open(path).convert("RGB"))
        h, w = image.shape[:2]
        mask_image = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_image)
        label = SOURCE / "Labels" / (path.stem + ".txt")
        for line in label.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            if fields[0] not in ("0", "1") or len(fields) < 7 or len(fields) % 2 != 1:
                raise ValueError(f"Unexpected YOLO polygon in {label}")
            values = [float(value) for value in fields[1:]]
            points = [(max(0, min(w - 1, round(x * w))), max(0, min(h - 1, round(y * h))))
                      for x, y in zip(values[::2], values[1::2])]
            draw.polygon(points, fill=1)
        mask = np.asarray(mask_image)
        y = random.randrange(max(1, h - PATCH + 1))
        x = random.randrange(max(1, w - PATCH + 1))
        return augment(image[y:y + PATCH, x:x + PATCH].copy(),
                       mask[y:y + PATCH, x:x + PATCH].copy())


def main() -> None:
    seed_everything(4201)
    os.environ.setdefault("TORCH_HOME", str((HERE.parent / "larvae_autoresearch" /
                                              "progress" / "torch_cache").resolve()))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset = ZenodoLarvae()
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0,
                        pin_memory=torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNet(classes=2, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    last = OUTPUT / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
    for epoch in range(start_epoch, 13):
        model.train()
        started, total = time.perf_counter(), 0.0
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(image)
                ce = F.cross_entropy(logits, target, ignore_index=255)
                valid = target != 255
                onehot = F.one_hot(target.masked_fill(~valid, 0), 2).permute(0, 3, 1, 2)
                probabilities = logits.softmax(1) * valid.unsqueeze(1)
                onehot = onehot * valid.unsqueeze(1)
                dice = 1 - ((2 * (probabilities * onehot).sum((0, 2, 3)) + 1e-6)
                            / ((probabilities + onehot).sum((0, 2, 3)) + 1e-6)).mean()
                loss = ce + dice
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        atomic_torch_save({"epoch": epoch, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, last)
        atomic_json({"epoch": epoch, "train_loss": total / len(loader),
                     "seconds": time.perf_counter() - started, "unique_original_stems": len(dataset)},
                    OUTPUT / "progress.json")
        print(json.dumps({"epoch": epoch, "loss": total / len(loader)}), flush=True)
    source_hash = hashlib.sha256((HERE / "external" / "BSF_Larvae_v1.zip").read_bytes()).hexdigest()
    atomic_torch_save({"encoder": model.encoder.cpu().state_dict(),
                       "source": "https://doi.org/10.5281/zenodo.13359376",
                       "license": "CC BY 4.0", "archive_sha256": source_hash,
                       "unique_original_stems": len(dataset), "epochs": 12},
                      OUTPUT / "encoder.pt")
    print("Saved external binary-pretrained encoder", flush=True)


if __name__ == "__main__":
    main()
