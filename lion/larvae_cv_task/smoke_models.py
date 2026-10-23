"""Check shape, gradients and memory of every architecture before long runs."""
from __future__ import annotations

import gc

import torch

from model import ARCHITECTURES, build_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in ARCHITECTURES:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_model(name, pretrained=False).to(device).train()
        inputs = torch.randn(2, 3, 512, 512, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(inputs)
            assert outputs.shape == (2, 3, 512, 512), (name, outputs.shape)
            loss = outputs.square().mean()
        loss.backward()
        assert all(parameter.grad is not None for parameter in model.parameters()), name
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
        print(f"{name}: parameters={sum(p.numel() for p in model.parameters()):,} peak_MiB={peak}", flush=True)
        del inputs, outputs, loss, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
