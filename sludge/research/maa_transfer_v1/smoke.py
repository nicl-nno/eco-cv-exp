"""CPU-only full Faster R-CNN integration probe on synthetic frames.

No real images, split manifests, calibration, validation or test evaluation.
Backward checks do not perform optimizer steps or create trained weights.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from adapter import FixedFrameMAADetector, verify_upstream

from sludge_micro.model import build_detector_raw, init_from_pretrained
from sludge_micro.runtime import package_digest

ROOT = Path(__file__).resolve().parents[2]


def probe():
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    digest = package_digest()
    upstream = verify_upstream()
    start = time.perf_counter()
    architecture = "fasterrcnn_resnet50_fpn_v2"
    model = build_detector_raw(
        architecture,
        8,
        128,
        160,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
    )
    weights = (
        ROOT / "outputs_updated_2026_10_02/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    )
    initialization = init_from_pretrained(model, weights, architecture)
    baseline_parameters = sum(p.numel() for p in model.parameters())
    model.eval()
    images = [torch.rand(3, 128, 160)]
    with torch.inference_mode():
        reference = model(images)[0]
    wrapped = FixedFrameMAADetector(model, image_size=(128, 160)).eval()
    with torch.inference_mode():
        identity = wrapped(images)[0]
    assert all(torch.equal(reference[k], identity[k]) for k in reference)
    print("COCO initialization + zero-gate full detector identity: OK", flush=True)

    with torch.no_grad():
        wrapped.detector.backbone.gate.fill_(0.1)
    wrapped.train()
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 40.0, 45.0], [75.0, 60.0, 120.0, 100.0]]),
            "labels": torch.tensor([1, 2]),
        }
    ]
    torch.manual_seed(20261023)
    losses = wrapped(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    block = wrapped.detector.backbone.attention.blocks[0]
    grads = {name: float(getattr(block, f"lambda_{name}").grad.abs()) for name in block.enable}
    assert all(value > 0 for value in grads.values())
    qkv_grad = block.attn.qkv.weight.grad
    assert torch.isfinite(qkv_grad).all() and qkv_grad.abs().sum() > 0
    print("Two-object torchvision loss + all six prior gradients: OK", flush=True)

    changed = [{"boxes": targets[0]["boxes"].clone(), "labels": torch.tensor([1, 3])}]
    changed[0]["boxes"][1] = torch.tensor([55.0, 55.0, 145.0, 115.0])
    wrapped.zero_grad(set_to_none=True)
    with torch.no_grad():
        torch.manual_seed(20261023)
        changed_losses = wrapped(images, changed)
        assert any(not torch.equal(losses[k], changed_losses[k]) for k in losses)
        empty_losses = wrapped(
            images, [{"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}]
        )
        assert all(torch.isfinite(value) for value in empty_losses.values())
    assert package_digest() == digest
    print("Second object affects loss; empty targets supported; package unchanged: OK", flush=True)
    return {
        "checked_at": datetime.now(UTC).isoformat(),
        "kind": "synthetic_cpu_integration_probe_not_quality_evaluation",
        "upstream_commit": upstream["commit"],
        "upstream_files_verified": len(upstream["files"]),
        "package_sha256_before_and_after": digest,
        "device": "cpu",
        "synthetic_frame": [128, 160],
        "synthetic_targets": 2,
        "real_images_read": 0,
        "test_reopened": False,
        "optimizer_steps": 0,
        "quality_evaluated": False,
        "coco_initialization": initialization,
        "zero_gate_identical_detections": True,
        "baseline_parameters": baseline_parameters,
        "maa_parameters": sum(p.numel() for p in wrapped.parameters()),
        "prior_lambda_abs_gradients": grads,
        "synthetic_losses": {k: float(v.detach()) for k, v in losses.items()},
        "second_object_affects_loss": True,
        "empty_targets_finite": True,
        "metadata": wrapped.export_metadata(),
        "elapsed_seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite an existing probe result; use a new path")
    result = probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
