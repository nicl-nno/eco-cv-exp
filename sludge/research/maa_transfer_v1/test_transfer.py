"""CPU-only technical checks, no real dataset or model-quality evaluation."""

import io
from collections import OrderedDict

import pytest
import torch
from research.maa_transfer_v1.adapter import (
    FixedFrameMAADetector,
    MAAP4Backbone,
    verify_upstream,
)
from torch import nn


@pytest.fixture(autouse=True)
def cpu_setup():
    torch.manual_seed(20261023)
    torch.set_num_threads(2)


class TinyFPN(nn.Module):
    out_channels = 256

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, 1)

    def forward(self, x):
        feature = self.conv(torch.nn.functional.adaptive_avg_pool2d(x, (4, 6)))
        return OrderedDict((str(i), feature.clone()) for i in range(4))


def bridge():
    return MAAP4Backbone(TinyFPN(), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).eval()


def test_upstream_snapshot_is_pinned():
    assert verify_upstream()["license"] == "MIT"


def test_missing_upstream_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="inventory"):
        verify_upstream(tmp_path)


def test_zero_gate_identity_and_p4_only():
    model = bridge()
    x = torch.randn(1, 3, 32, 64)
    with torch.no_grad():
        baseline = model.base(x)
        identity = model(x)
        for key in baseline:
            assert torch.equal(baseline[key], identity[key])
        model.gate.fill_(0.1)
        enabled = model(x)
    assert not torch.equal(baseline["2"], enabled["2"])
    for key in ("0", "1", "3"):
        assert torch.equal(baseline[key], enabled[key])


def test_prior_receives_unnormalized_rgb():
    model = bridge()
    rgb = torch.rand(1, 3, 32, 64)
    captured = []
    hook = model.attention.register_forward_pre_hook(lambda _, args: captured.append(args[1]))
    with torch.no_grad():
        model((rgb - model.image_mean) / model.image_std)
    hook.remove()
    torch.testing.assert_close(captured[0], rgb)


def test_gradients_and_state_roundtrip():
    model = bridge()
    model.gate.data.fill_(0.1)
    x = torch.randn(1, 3, 32, 64)
    model(x)["2"].square().mean().backward()
    block = model.attention.blocks[0]
    for parameter in [model.gate, block.attn.qkv.weight] + [
        getattr(block, f"lambda_{name}") for name in block.enable
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = bridge()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(x)["2"], restored(x)["2"], rtol=0, atol=0)


def test_density_default_is_not_image_content_dependent():
    bridge()  # pin and import upstream
    from maa.attention.maa.density import DensityPrior

    prior = DensityPrior()
    a = prior(torch.zeros(1, 3, 32, 32), 16, 4, 4)
    b = prior(torch.rand(1, 3, 32, 32), 16, 4, 4)
    assert torch.equal(a, b)


def test_upstream_rectangular_grid_limitation():
    bridge()
    from maa.attention.common import build_token_grid

    grid = build_token_grid(12, torch.device("cpu"), torch.float32)
    assert torch.equal(grid[:, 0], torch.zeros(12))
    # v1 deliberately uses the upstream square 16x16 token bridge instead.


def test_upstream_detection_loss_ignores_second_object():
    bridge()
    from maa.models.detection.internimage_h import InternImageHDetector

    model = InternImageHDetector(attention="none")
    logits, boxes = torch.randn(1, 4, 2, 2), torch.randn(1, 4, 2, 2)
    a = {"labels": torch.tensor([[1, 2]]), "boxes": torch.rand(1, 2, 4)}
    b = {"labels": torch.tensor([[1, 3]]), "boxes": a["boxes"].clone()}
    b["boxes"][:, 1] += 10
    la, lb = model.compute_losses(logits, boxes, a), model.compute_losses(logits, boxes, b)
    assert all(torch.equal(la[k], lb[k]) for k in la)


def test_upstream_deformable_offsets_do_not_change_output():
    bridge()
    from maa.models.detection.internimage_h import DeformableConv2d

    layer = DeformableConv2d(4).eval()
    x = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        before = layer(x)
        layer.offset.weight.add_(100)
        layer.offset.bias.add_(100)
        after = layer(x)
    assert torch.equal(before, after)


def test_frame_guards_and_frozen_bn():
    from sludge_micro.model import build_detector_raw

    detector = build_detector_raw("fasterrcnn_resnet50_fpn_v2", 8, 64, 96)
    model = FixedFrameMAADetector(detector, image_size=(64, 96)).train()
    assert all(not m.training for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    with pytest.raises(ValueError, match="unpadded"):
        model([torch.rand(3, 64, 95)])
    with pytest.raises(ValueError, match="unpadded"):
        model([torch.rand(3, 64, 96)] * 2)
    with pytest.raises(ValueError, match="finite"):
        model([torch.full((3, 64, 96), float("nan"))])


def test_no_prior_capacity_control():
    model = MAAP4Backbone(TinyFPN(), [0, 0, 0], [1, 1, 1], priors_enabled=False)
    assert not any(model.attention.blocks[0].enable.values())
    assert sum(p.numel() for p in model.attention.parameters()) == sum(
        p.numel() for p in bridge().attention.parameters()
    )
