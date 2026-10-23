"""FLOPs, parameter count, and inference latency helpers."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
import torch.nn as nn

from maa.constants import INPUT_SIZE
from maa.training.common import count_params

logger = logging.getLogger(__name__)


def estimate_flops(
    model: nn.Module,
    input_size: tuple[int, ...] = (1, 3, INPUT_SIZE, INPUT_SIZE),
    device: torch.device | None = None,
) -> float | None:
    """
    Estimate GFLOPs using torch.profiler when available; returns None otherwise.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size, device=device)

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            with_flops=True,
        ) as prof:
            with torch.inference_mode():
                model(dummy)
        total = sum(e.flops for e in prof.key_averages() if e.flops is not None)
        return float(total) / 1e9
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.debug("FLOPs estimation unavailable: %s", exc)
        return None


def benchmark_inference(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, float]:
    """Warmup + timed inference returning latency in milliseconds."""
    model.eval()
    device = sample_input.device
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    return {"latency_ms": elapsed_ms}


def profile_model(
    model: nn.Module,
    *,
    input_size: tuple[int, ...] = (1, 3, INPUT_SIZE, INPUT_SIZE),
    device: torch.device | None = None,
    warmup: int = 10,
    repeats: int = 50,
) -> dict[str, Any]:
    """Combined params / FLOPs / latency report."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy = torch.randn(*input_size, device=device)
    latency = benchmark_inference(model, dummy, warmup=warmup, repeats=repeats)
    flops = estimate_flops(model, input_size=input_size, device=device)
    report: dict[str, Any] = {
        "params": count_params(model),
        "latency_ms": latency["latency_ms"],
        "input_size": input_size[-1],
    }
    if flops is not None:
        report["gflops"] = flops
    return report
