"""Selection, fixed metrics and scheduling safety for the development CV stage."""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    folder = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("cv_runner_test", folder / "run_grouped_cv.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.night, "OUTPUTS", tmp_path)
    monkeypatch.setattr(
        module.night, "now", lambda: datetime.fromisoformat("2026-10-21T03:00:00+03:00")
    )
    return module


def measurements(runner):
    return [
        {
            "recipe": recipe,
            "seed": seed,
            "macro_f1": {"control15": 0.3, "rotation15": 0.2, "rotation30": 0.31, "cosine30": 0.34}[
                recipe
            ],
        }
        for recipe in runner.night.RECIPES
        for seed in runner.night.SEEDS
    ]


def test_selection_uses_all_seeds_not_best_individual_run(runner):
    rows = measurements(runner)
    one = next(r for r in rows if r["recipe"] == "rotation15")
    one["macro_f1"] = 0.6
    selected, means = runner.select_candidate(rows)
    assert selected == "cosine30"
    assert means["rotation15"] == pytest.approx(1.0 / 3)


def test_selection_refuses_partial_or_duplicate_seed_pairs(runner):
    rows = measurements(runner)
    with pytest.raises(ValueError, match="twelve unique"):
        runner.select_candidate(rows[:-1])
    with pytest.raises(ValueError, match="twelve unique"):
        runner.select_candidate([*rows[:-1], rows[0]])


def test_no_cv_if_candidate_does_not_beat_control(runner):
    rows = [{**r, "macro_f1": 0.3} for r in measurements(runner)]
    assert runner.select_candidate(rows)[0] is None


def test_candidate_ties_are_alphabetical(runner):
    rows = [
        {**r, "macro_f1": 0.3 if r["recipe"] == "control15" else 0.4} for r in measurements(runner)
    ]
    assert runner.select_candidate(rows)[0] == "cosine30"


def test_nonfinite_metrics_are_rejected(runner):
    rows = measurements(runner)
    rows[0]["macro_f1"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        runner.select_candidate(rows)


def test_original_queue_must_finish_before_selection(runner):
    (runner.night.OUTPUTS / "queue_state.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError, match="has not finished"):
        runner.first_stage_analysis()


def test_shared_supervisor_lock_blocks_second_gpu_stage(runner):
    with (
        runner.night.FileLock(runner.night.OUTPUTS / "queue.lock"),
        pytest.raises(RuntimeError, match="Another overnight process"),
    ):
        runner.run()


def test_deadline_refuses_new_cv_stage(runner, monkeypatch):
    monkeypatch.setattr(
        runner.night, "now", lambda: datetime.fromisoformat("2026-10-21T08:01:00+03:00")
    )
    with pytest.raises(RuntimeError, match="Night deadline reached"):
        runner.run()


def test_fixed_threshold_and_zero_detection_frames_are_preserved(runner, config):
    name = config.classes.names[0]
    boxes = {"image_id": "a", "class_name": name, "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}
    annotations = pd.DataFrame([{**boxes, "supervision": "target"}])
    predictions = pd.DataFrame([{**boxes, "score": 0.19}])
    result = runner.fixed_metrics(config, predictions, annotations, ["a", "empty"])
    assert result["confidence"] == 0.2
    assert result["images"] == result["counting"]["images"] == 2
    assert result["classes"][0]["tp"] == 0
    assert result["counting"]["classes"][0]["mae_per_frame"] == 0.5
    predictions["score"] = 0.21
    assert runner.fixed_metrics(config, predictions, annotations, ["a", "empty"])["macro_f1"] == 1.0
    with pytest.raises(ValueError, match="Duplicate validation"):
        runner.fixed_metrics(config, predictions, annotations, ["a", "a"])
    with pytest.raises(ValueError, match="contamination"):
        runner.fixed_metrics(config, predictions, annotations, ["other"])
