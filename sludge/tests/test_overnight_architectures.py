"""Synthetic third-night architecture tests, no scientific training or test data."""

import importlib
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sludge_micro.dataset import Sample


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return (
        importlib.import_module("architecture_components"),
        importlib.import_module("overnight_architectures"),
    )


def sample(labels=(1, 2, 3)):
    return Sample(
        "fit",
        Path("unused.bmp"),
        100,
        100,
        tuple((i * 20.0, 0.0, i * 20.0 + 10, 10.0) for i in range(len(labels))),
        labels,
    )


def pred(name="attached_ciliates", score=0.8, x=0.0, image_id="fit"):
    return {
        "image_id": image_id,
        "class_name": name,
        "score": score,
        "x1": x,
        "y1": 0.0,
        "x2": x + 10,
        "y2": 10.0,
    }


def test_remap_keeps_specialist_negatives_and_originals(modules, config):
    c, _ = modules
    s = sample()
    assert c.mapping(config.classes.names, "agnostic") == dict.fromkeys(range(1, 8), 1)
    labels = c.mapping(config.classes.names, "filamentous")
    assert labels == {2: 1}
    result = c.remap_samples([s, sample((1,))], labels)
    assert result[0].labels == (1,)
    assert result[0].boxes == (s.boxes[1],)
    assert result[1].labels == result[1].boxes == ()
    assert s.labels == (1, 2, 3)
    assert c.mapping(config.classes.names, "others") == {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}


def test_remap_rejects_ignore(modules):
    c, _ = modules
    with pytest.raises(ValueError, match="ignore"):
        c.remap_samples([sample((-1,))], {1: 1})


def test_crop_manifest_fit_only_caps_and_jitter(modules):
    c, _ = modules
    s = sample((1,))
    predictions = pd.DataFrame(
        [*[pred() for _ in range(5)], *[pred(x=50) for _ in range(8)], pred(x=5)]
    )
    result = c.crop_manifest([s], predictions, 123)
    assert result.source.value_counts().to_dict() == {
        "fit_proposal_negative": 5,
        "fit_proposal_positive": 2,
        "gt": 1,
        "jitter": 1,
    }
    assert set(result.loc[result.source == "fit_proposal_negative", "label"]) == {0}
    pd.testing.assert_frame_equal(result, c.crop_manifest([s], predictions, 123))
    with pytest.raises(ValueError, match="contamination"):
        c.crop_manifest([s], predictions.assign(image_id="outer"), 123)
    with pytest.raises(ValueError, match="duplicate"):
        c.crop_manifest([s, s], predictions, 123)
    with pytest.raises(ValueError, match="ignore"):
        c.crop_manifest([sample((-1,))], predictions, 123)


def test_crop_manifest_empty_gt_and_boundaries(modules):
    c, _ = modules
    predictions = pd.DataFrame([pred(x=50)])
    result = c.crop_manifest([sample(())], predictions, 123)
    assert result.label.tolist() == [0]
    assert c.crop_weights([0] * 100 + [1] * 10 + [2])[-1] == 3
    assert c.crop_weights([0] * 100 + [1] * 10 + [2])[0] == 1


def test_crop_rgb_square_normalization(modules):
    c, _ = modules
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    tensor = c.crop_tensor(image, [0, 0, 40, 20])
    assert tensor.shape == (3, 224, 224)
    assert tensor[0, 112, 112] == 1
    assert tensor[1:, 112, 112].sum() == 0
    assert tensor[:, 0, :].sum() == 0
    assert torch.isfinite(c.normalize(tensor[None])).all()


def test_crop_dataset_deterministic_epoch_augmentation(modules, monkeypatch):
    c, _ = modules
    image = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    monkeypatch.setattr(c, "read_bgr", lambda _: image)
    manifest = pd.DataFrame([{**pred(), "label": 1}])
    dataset = c.CropDataset([sample()], manifest, 32)
    first, _ = dataset[0]
    assert torch.equal(first, dataset[0][0])
    dataset.epoch = 1
    assert not torch.equal(first, dataset[0][0])
    with pytest.raises(ValueError, match="contamination"):
        c.CropDataset([sample()], manifest.assign(image_id="outer"), 32)


def test_cascade_background_score_product_and_nms(modules):
    c, _ = modules
    proposals = pd.DataFrame([pred(score=0.9), pred(score=0.8), pred(score=0.7), pred(x=30)])
    probabilities = np.array(
        [[0.01, 0.89, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.9, 0.05, 0.05]]
    )
    result = c.cascade_rows("fit", proposals, probabilities, ["a", "b"])
    assert result.class_name.tolist() == ["a", "b"]
    assert result.score.tolist() == pytest.approx([0.801, 0.56])
    assert c.cascade_rows("fit", proposals.iloc[:0], np.array([]), ["a", "b"]).empty


def test_specialist_cross_suppression_only_and_tie_order(modules):
    c, _ = modules
    first = pd.DataFrame([pred("filamentous_bacteria", score=0.8)])
    second = pd.DataFrame([pred(score=0.8), pred(x=4), pred(x=40), pred(x=40, score=0.7)])
    merged = c.merge_specialists(first, second)
    assert len(merged) == 4
    assert merged.iloc[0].class_name == "filamentous_bacteria"
    assert sum(merged.x1 == 40) == 2  # no extra within-specialist suppression
    assert len(c.merge_specialists(first, second, False)) == 5
    assert len(c.merge_specialists(first, pd.DataFrame([pred(score=0.9)]))) == 1
    assert (
        c.merge_specialists(first, pd.DataFrame([pred(score=0.9)])).iloc[0].class_name
        == "attached_ciliates"
    )


def test_merge_global_cap_and_empty(modules):
    c, _ = modules
    first = pd.DataFrame([pred(x=20 * i) for i in range(150)])
    empty = first.iloc[:0]
    assert len(c.merge_specialists(first, empty)) == 100
    assert c.merge_specialists(empty, empty).columns.tolist() == c.PREDICTION_COLUMNS


def test_diagnostics_distinguish_localization_class_duplicate_ignore(modules, config):
    c, _ = modules
    annotations = pd.DataFrame(
        [{**pred(), "supervision": "target"}, {**pred(x=50), "supervision": "ignore"}]
    )
    predictions = pd.DataFrame(
        [pred("rotifers", score=0.9), pred(score=0.8), pred(x=50), pred(x=80)]
    )
    result = c.diagnostics(config, predictions, annotations)
    assert result["localization_tp"] == 1
    assert result["localization_fp"] == 2
    assert result["duplicate_fp"] == 1
    assert result["ignored"] == 1
    assert result["matched_class_accuracy"] == 0
    assert result["localization_recall"] == 1
    assert c.diagnostics(config, predictions.iloc[:0], annotations)["localization_fn"] == 1


def test_calibration_barrier_and_freeze(modules, tmp_path, monkeypatch):
    _, r = modules
    monkeypatch.setattr(r.old, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="before outer"):
        r.require_calibration(tmp_path, ["cascade"])
    path = tmp_path / "calibration_cascade.json"
    r.freeze_json(path, {"threshold": 0.2})
    r.freeze_json(path, {"threshold": 0.2})
    assert len(r.require_calibration(tmp_path, ["cascade"])) == 1
    with pytest.raises(ValueError, match="frozen"):
        r.freeze_json(path, {"threshold": 0.3})


def test_deadline_precedes_registration_and_lock(modules, monkeypatch):
    c, r = modules
    monkeypatch.setattr(r.night, "now", lambda: r.DEADLINE + timedelta(seconds=1))
    with pytest.raises(RuntimeError, match="Deadline"):
        r.run()
    with pytest.raises(RuntimeError, match="Deadline"):
        c.check_deadline(r.DEADLINE)


def test_merge_counts_distinguish_cap_from_suppression(modules):
    c, _ = modules
    first = pd.DataFrame([pred("filamentous_bacteria")])
    second = pd.DataFrame([pred(score=0.7), *[pred(x=20 * i) for i in range(1, 151)]])
    result, counts = c.merge_specialists(first, second, return_counts=True)
    assert counts == {"input": 152, "cross_suppressed": 1, "top100_discarded": 51, "output": 100}
    assert len(result) == 100


def test_atomic_parquet_overwrites_only_at_commit(modules, tmp_path):
    c, _ = modules
    path = tmp_path / "predictions.parquet"
    frame = pd.DataFrame([pred()])
    c.atomic_parquet(path, frame)
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    assert not path.with_name(path.name + ".part").exists()


def test_queue_budget_identity_and_failed_retry(modules):
    _, r = modules
    jobs = [{"index": i} for i in range(12)]
    reg = {"jobs": jobs}
    queue = [{**j, "status": "pending"} for j in jobs]
    r.validate_queue(reg, queue)
    queue[0]["status"] = "failed"
    with pytest.raises(RuntimeError, match="no blind retry"):
        r.validate_queue(reg, queue)
    queue[0]["status"] = "pending"
    queue[0]["index"] = 99
    with pytest.raises(ValueError, match="differs"):
        r.validate_queue(reg, queue)


def test_shared_os_lock_exclusion(modules, tmp_path):
    _, r = modules
    path = tmp_path / "gpu.lock"
    with r.night.FileLock(path):
        with pytest.raises(RuntimeError, match="holds this lock"):
            with r.night.FileLock(path):
                pass
    with r.night.FileLock(path):
        pass


def test_eligibility_does_not_choose_union(modules):
    _, r = modules
    names = ["attached_ciliates", "filamentous_bacteria", "rotifers"]

    def rows(f1, mae):
        return {
            "count_guarded": {
                "mean_macro_f1": f1,
                "classes": [{"name": n, "mean_f1": f1, "mean_mae_per_frame": mae} for n in names],
            }
        }

    means = {
        "cosine30": rows(0.4, 1),
        "cascade": rows(0.4, 0.89),
        "specialists": rows(0.5, 1),
        "specialists_union": rows(0.8, 0.2),
    }
    result = r.eligibility(means)
    assert result["cascade"]["eligible"]
    assert not result["specialists"]["eligible"]
    assert "specialists_union" not in result


@pytest.mark.slow
def test_local_coco_classifier_backbone_and_cpu_step(modules):
    c, r = modules
    checkpoint = (
        r.ROOT
        / "outputs_updated_2026_10_02/pretrained/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    )
    if not checkpoint.exists():
        pytest.skip("Local COCO weights unavailable")
    torch.set_num_threads(4)
    model = c.build_classifier(checkpoint, 7)
    assert model.fc.out_features == 8
    assert all(
        n.startswith(("layer4.", "fc.")) for n, p in model.named_parameters() if p.requires_grad
    )
    start = c.batchnorm_statistics(model)
    c.set_training_mode(model, "frozen")
    loss = torch.nn.functional.cross_entropy(
        model(c.normalize(torch.zeros(1, 3, 224, 224))), torch.tensor([1])
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.fc.weight.grad is not None
    assert model.conv1.weight.grad is None
    assert c.batchnorm_check(model, start, "frozen")["changed"] == 0


def test_classifier_adamw_resume_exact_cpu(modules, tmp_path):
    c, _ = modules
    generator = torch.Generator().manual_seed(5)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x = torch.ones(2, 3)
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    path = tmp_path / "resume.pt"
    c.save_resume_state(path, model, optimizer, generator, [{"epoch": 1, "loss": 1.0}])
    state = torch.load(path, weights_only=False)
    model(x).sum().backward()
    optimizer.step()
    expected = model.weight.detach().clone()
    replica = torch.nn.Linear(3, 2)
    replica_optimizer = torch.optim.AdamW(replica.parameters(), lr=10.0)
    history = c.restore_resume_state(state, replica, replica_optimizer, generator)
    assert len(history) == 1
    replica(x).sum().backward()
    replica_optimizer.step()
    assert torch.equal(replica.weight, expected)
