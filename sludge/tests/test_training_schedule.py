"""Small CPU checks for scheduling, accumulation and resume RNG state."""

import pytest
import torch

from sludge_micro.train import (
    epoch_learning_rate,
    restore_resume_state,
    save_resume_state,
    train_epoch,
)


def test_constant_schedule_preserves_recipe(config):
    assert all(epoch_learning_rate(config.train, e) == config.train.lr for e in range(15))


def test_cosine_schedule_and_warmup(config):
    train = config.train.model_copy(
        update={"epochs": 8, "warmup_epochs": 2, "lr_schedule": "cosine", "final_lr_factor": 0.1}
    )
    rates = [epoch_learning_rate(train, e) for e in range(8)]
    assert rates[:3] == [train.lr / 2, train.lr, train.lr]
    assert rates[-1] == pytest.approx(train.lr * 0.1)
    assert rates[2:] == sorted(rates[2:], reverse=True)


class LinearLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets):
        return {"loss": self.weight * images[0].sum()}


def test_partial_accumulation_window_is_not_underweighted():
    model = LinearLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = ([torch.tensor([2.0])], [{}])
    train_epoch(model, [batch] * 3, optimizer, 2, torch.device("cpu"))
    # Two updates, each the average gradient 2, including the one-batch tail.
    assert model.weight.item() == pytest.approx(0.6)


def test_resume_restores_cpu_and_cuda_rng(tmp_path):
    model = LinearLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    generator = torch.Generator().manual_seed(17)
    path = tmp_path / "resume.pt"
    save_resume_state(path, model, optimizer, generator, [])
    expected = torch.rand(4)
    cuda_expected = torch.rand(4, device="cuda") if torch.cuda.is_available() else None
    state = torch.load(path, weights_only=False)
    restore_resume_state(state, model, optimizer, generator)
    assert torch.equal(torch.rand(4), expected)
    if cuda_expected is not None:
        assert torch.equal(torch.rand(4, device="cuda"), cuda_expected)


def test_invalid_warmup_is_rejected(config):
    with pytest.raises(ValueError, match="warmup_epochs"):
        type(config.train).model_validate(config.train.model_dump() | {"warmup_epochs": 100})
