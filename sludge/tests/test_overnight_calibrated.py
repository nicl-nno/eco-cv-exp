"""Synthetic checks for second-night partitions, thresholds and queue safety."""

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    folder = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "calibrated_test", folder / "overnight_calibrated.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.night, "OUTPUTS", tmp_path / "locks")
    monkeypatch.setattr(module, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(module, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(
        module.night, "now", lambda: datetime.fromisoformat("2026-10-21T23:00:00+03:00")
    )
    return module


def frames():
    return pd.DataFrame(
        [
            {
                "image_id": str(i),
                "group": f"g{i}",
                "split": "train",
                "pixel_sha256": f"p{i}",
                "file_sha256": f"f{i}",
                "dhash": f"{i:064x}",
            }
            for i in range(10)
        ]
    )


def box(image_id, name="attached_ciliates", x=0):
    return {
        "image_id": str(image_id),
        "class_name": name,
        "x1": x,
        "y1": 0.0,
        "x2": x + 10.0,
        "y2": 10.0,
    }


def synthetic_calibration():
    manifest = frames().assign(split="validation")
    anns = pd.DataFrame([{**box(i), "supervision": "target"} for i in range(6)])
    predictions = pd.DataFrame(
        [
            *[{**box(i), "score": 0.8} for i in range(6)],
            *[{**box(i, x=20), "score": 0.2} for i in range(10)],
        ]
    )
    return manifest, anns, predictions


def test_partition_deterministic_and_rare_fit_class_retained(runner):
    manifest = frames()
    anns = pd.DataFrame(
        [{**box(i, "rare" if i == 0 else "common"), "supervision": "target"} for i in range(10)]
    )
    result = runner.partition_inner(manifest, anns, ("common", "rare"), 17)
    pd.testing.assert_frame_equal(
        result, runner.partition_inner(manifest, anns, ("common", "rare"), 17)
    )
    assert result.loc[result.image_id == "0", "split"].item() == "train"
    assert result.groupby("group").split.nunique().max() == 1
    assert set(result.split) == {"train", "validation"}
    runner.check_inner(result, manifest, 0)
    with pytest.raises(ValueError, match="non-training"):
        runner.partition_inner(
            manifest,
            pd.concat([anns, anns.iloc[:1].assign(image_id="test")]),
            ("common", "rare"),
            17,
        )


def test_inner_identity_membership_and_near_duplicate_guards(runner):
    original = frames()
    inner = original.copy()
    inner.loc[0, "split"] = "validation"
    runner.check_inner(inner, original, 0)
    with pytest.raises(ValueError, match="near-duplicate"):
        runner.check_inner(inner, original, 1)
    with pytest.raises(ValueError, match="membership"):
        runner.check_inner(inner.iloc[1:], original, 0)
    with pytest.raises(ValueError, match="identity"):
        runner.check_inner(inner.assign(group="altered"), original, 0)


def test_thresholds_use_calibration_only_and_count_zero_frames(runner, config):
    manifest, anns, predictions = synthetic_calibration()
    selected = runner.calibrate(config, predictions, anns, manifest)
    first = config.classes.names[0]
    assert selected["policies"]["global_f1"][first] == 0.8
    assert selected["policies"]["count_guarded"][first] == 0.8
    assert selected["class_details"][first]["groups"] == 6
    assert selected["class_details"][first]["reason"] == "class_count_guarded"
    assert selected["class_details"][config.classes.names[1]]["reason"].endswith(
        "insufficient_support"
    )
    filtered = runner.apply_thresholds(predictions, selected["policies"]["count_guarded"])
    metrics = runner.measure(config, filtered, anns, manifest.image_id.tolist())
    assert metrics["images"] == 10
    assert metrics["macro_f1"] == 1.0
    assert metrics["counting"]["classes"][0]["mae_per_frame"] == 0.0
    assert metrics["confidence"] is None
    for polluted in (predictions.assign(image_id="outer"),):
        with pytest.raises(ValueError, match="contamination"):
            runner.calibrate(config, polluted, anns, manifest)
    with pytest.raises(ValueError, match="annotation contamination"):
        runner.calibrate(config, predictions, anns.assign(image_id="test"), manifest)
    with pytest.raises(ValueError, match="calibration-only"):
        runner.calibrate(config, predictions, anns, manifest.assign(split="test"))


def test_support_fallback_counts_groups_not_frames(runner, config):
    manifest, anns, predictions = synthetic_calibration()
    selected = runner.calibrate(config, predictions, anns, manifest.assign(group="one"))
    assert selected["class_details"][config.classes.names[0]]["reason"].endswith(
        "insufficient_support"
    )
    selected = runner.calibrate(config, predictions.iloc[:0], anns, manifest)
    assert selected["class_details"][config.classes.names[0]]["reason"].endswith(
        "no_true_positives"
    )
    empty = runner.calibrate(config, predictions.iloc[:0], anns.iloc[:0], manifest)
    assert set(empty["policies"]["global_f1"].values()) == {0.2}


def test_measure_rejects_duplicate_oof_and_contamination(runner, config):
    manifest, anns, predictions = synthetic_calibration()
    with pytest.raises(ValueError, match="duplicate frame"):
        runner.measure(config, predictions, anns, [*manifest.image_id, "0"])
    with pytest.raises(ValueError, match="annotation contamination"):
        runner.measure(config, predictions, anns.assign(image_id="test"), list(manifest.image_id))
    with pytest.raises(ValueError, match="no frozen threshold"):
        runner.apply_thresholds(predictions, {})


@pytest.mark.parametrize("lock", ["queue.lock", "gpu.lock"])
def test_shared_locks_prevent_second_queue(runner, lock):
    with (
        runner.night.FileLock(runner.night.OUTPUTS / lock),
        pytest.raises(RuntimeError, match="Another overnight process"),
    ):
        runner.run()


def test_deadline_blocks_prepare_and_run(runner, monkeypatch):
    monkeypatch.setattr(runner.night, "now", lambda: runner.DEADLINE)
    for function in (runner.run, runner.prepare):
        with pytest.raises(RuntimeError, match=r"[Dd]eadline"):
            function()


def test_partial_summary_never_claims_mean_or_eligibility(runner):
    runner.summarize({"jobs": [], "development_images": 10}, [])
    result = runner.read_json(runner.REPORTS / "summary.json")
    assert result["completed_jobs"] == 0
    assert result["means"] == result["eligibility"] == {}


def test_count_guard_minimizes_mae_within_f1_tolerance(runner, config, monkeypatch):
    manifest, anns, predictions = synthetic_calibration()
    name = config.classes.names[0]
    monkeypatch.setattr(runner, "confidence_grid", lambda _: [0.2, 0.4, 0.6])

    def metric(det, ref, classes, threshold):
        f1 = {0.2: 1.0, 0.4: 0.91, 0.6: 0.89}[threshold]
        return {
            "confidence": threshold,
            "macro_f1": f1,
            "classes": [{"name": c, "f1": f1 if c == name else None} for c in classes],
        }

    def counts(det, ref, ids, classes, threshold):
        return {
            "classes": [
                {"name": c, "mae_per_frame": {0.2: 2.0, 0.4: 1.0, 0.6: 0.0}[threshold]}
                for c in classes
            ]
        }

    monkeypatch.setattr(runner, "summarise", metric)
    monkeypatch.setattr(runner, "count_errors", counts)
    result = runner.calibrate(config, predictions, anns, manifest)
    assert result["policies"]["global_f1"][name] == 0.2
    assert result["policies"]["count_guarded"][name] == 0.4


def test_supervisor_deadline_stops_own_worker_and_keeps_pending_jobs(runner, monkeypatch):
    from unittest.mock import Mock

    jobs = [{"fold": i // 8, "run_id": f"r{i}", "config": {"path": "fake"}} for i in range(24)]
    monkeypatch.setattr(runner, "prepare", lambda: {"jobs": jobs})
    monkeypatch.setattr(runner, "load_config", lambda _: None)
    monkeypatch.setattr(runner, "run_dirs", lambda *args: (runner.REPORTS, runner.OUTPUTS))
    monkeypatch.setattr(runner, "summarize", lambda *args: None)
    process = Mock(pid=123)

    def timeout(**kwargs):
        assert kwargs["timeout"] > 0
        monkeypatch.setattr(runner.night, "now", lambda: runner.DEADLINE)
        raise runner.subprocess.TimeoutExpired("synthetic", 1)

    process.wait.side_effect = timeout
    launch, stop = Mock(return_value=process), Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    monkeypatch.setattr(runner.night, "stop_worker", stop)
    runner.run()
    launch.assert_called_once()
    stop.assert_called_once_with(process)
    state = runner.read_json(runner.OUTPUTS / "queue_state.json")
    assert state["status"] == "deadline_reached"
    assert state["queue"][0]["status"] == "deadline_stopped"
    assert all(j["status"] == "pending" for j in state["queue"][1:])


def test_failed_queue_requires_inspection_before_restart(runner, monkeypatch):
    jobs = [{"run_id": f"r{i}"} for i in range(24)]
    runner.night.atomic_json(
        runner.OUTPUTS / "queue_state.json", {"queue": [{**j, "status": "failed"} for j in jobs]}
    )
    monkeypatch.setattr(runner, "prepare", lambda: {"jobs": jobs})
    with pytest.raises(RuntimeError, match="explicit inspection"):
        runner.run()
