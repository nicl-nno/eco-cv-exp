"""Synthetic checks for selected-class routing, frozen calibration and budgets."""

import importlib
from pathlib import Path

import pandas as pd
import pytest

from sludge_micro.dataset import Sample


@pytest.fixture
def hybrid(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("overnight_hybrids")


def pred(name, score=0.8, image_id="cal", x=0):
    return {
        "image_id": image_id,
        "class_name": name,
        "score": score,
        "x1": x,
        "y1": 0.0,
        "x2": x + 10.0,
        "y2": 10.0,
    }


def test_finite_design_budget(hybrid, config):
    assert len(hybrid.GROUPS) * len(hybrid.SEEDS) * 3 == 30
    assert len(hybrid.REPLACEMENTS) == 12
    for method in hybrid.REPLACEMENTS:
        route = hybrid.routes(config.classes.names, method)
        assert set(route) == set(config.classes.names)
        assert 1 <= len(set(route.values())) <= 4
        assert all(
            route[n] == "general"
            for n in config.classes.names
            if n not in hybrid.REPLACEMENTS[method]
        )


def test_subset_mapping_stable_contiguous(hybrid, config):
    assert hybrid.label_map(config.classes.names, hybrid.RARE) == {5: 1, 6: 2, 7: 3}
    assert hybrid.label_map(config.classes.names, (hybrid.FILAMENT, hybrid.ATTACHED)) == {
        1: 1,
        2: 2,
    }
    for invalid in [(), ("unknown",), (hybrid.ATTACHED, hybrid.ATTACHED)]:
        with pytest.raises(ValueError):
            hybrid.label_map(config.classes.names, invalid)


def test_subset_remap_preserves_empty_fit_frames(hybrid, config):
    samples = [Sample("fit", Path("unused"), 20, 20, ((0, 0, 10, 10),), (2,))]
    remapped = hybrid.components.remap_samples(
        samples, hybrid.label_map(config.classes.names, (hybrid.ATTACHED,))
    )
    assert len(remapped) == 1 and remapped[0].labels == ()
    assert samples[0].labels == (2,)


def test_route_replaces_not_unions_and_preserves_other_classes(hybrid, config):
    general = pd.DataFrame([pred(n) for n in config.classes.names])
    specialist = pd.DataFrame([pred(hybrid.ATTACHED, score=0.3, x=30)])
    route = hybrid.routes(config.classes.names, "h_attached")
    combined = hybrid.combine({"general": general, "attached": specialist}, route)
    assert combined[combined.class_name == hybrid.ATTACHED].score.tolist() == [0.3]
    pd.testing.assert_frame_equal(
        combined[combined.class_name != hybrid.ATTACHED].reset_index(drop=True),
        general[general.class_name != hybrid.ATTACHED][
            hybrid.components.PREDICTION_COLUMNS
        ].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(general, pd.DataFrame([pred(n) for n in config.classes.names]))


def test_no_global_cap_or_cross_class_suppression(hybrid):
    a = pd.DataFrame([pred("a", x=0) for _ in range(100)])
    b = pd.DataFrame([pred("b", score=0.99, x=0) for _ in range(100)])
    result = hybrid.combine({"one": a, "two": b}, {"a": "one", "b": "two"})
    assert len(result) == 200
    with pytest.raises(ValueError, match="Missing"):
        hybrid.combine({"one": a}, {"b": "two"})


def test_primary_thresholds_are_source_preserving(hybrid):
    selections = {
        "general": {"policies": {"count_guarded": {"a": 0.2, "b": 0.7}}},
        "specialist": {"policies": {"count_guarded": {"a": 0.9}}},
    }
    assert hybrid.routed_thresholds(selections, {"a": "specialist", "b": "general"}) == {
        "a": 0.9,
        "b": 0.7,
    }


def test_subset_calibration_retains_ignore(hybrid):
    anns = pd.DataFrame(
        [
            {"class_name": "a", "supervision": "target"},
            {"class_name": "b", "supervision": "target"},
            {"class_name": "unknown", "supervision": "ignore"},
        ]
    )
    assert hybrid.subset_annotations(anns, ["a"]).class_name.tolist() == ["a", "unknown"]


def test_stage_membership_rejects_outer_or_unknown(hybrid):
    with pytest.raises(ValueError, match="contamination"):
        hybrid.assert_membership(pd.DataFrame([pred("a", image_id="outer")]), ["cal"], ["a"])
    with pytest.raises(ValueError, match="contamination"):
        hybrid.assert_membership(pd.DataFrame([pred("bad")]), ["cal"], ["a"])


def test_freeze_does_not_overwrite_and_requires_all_before_outer(hybrid, tmp_path):
    hybrid.previous.freeze_json(tmp_path / "calibration_a.json", {"policies": {}})
    with pytest.raises(ValueError, match="frozen"):
        hybrid.previous.freeze_json(tmp_path / "calibration_a.json", {"policies": {"changed": 0.4}})
    with pytest.raises(ValueError, match="before outer"):
        hybrid.previous.require_calibration(tmp_path, ["a", "b"])


def test_queue_mismatch_and_failed_no_retry(hybrid):
    jobs = [{"index": i} for i in range(6)]
    queue = [{**j, "status": "pending"} for j in jobs]
    hybrid.validate_queue({"jobs": jobs}, queue)
    queue[0]["status"] = "failed"
    with pytest.raises(RuntimeError, match="no blind retry"):
        hybrid.validate_queue({"jobs": jobs}, queue)
    queue[0]["index"] = 99
    with pytest.raises(ValueError, match="registration"):
        hybrid.validate_queue({"jobs": jobs}, queue)


def test_strict_eligibility_does_not_relax_for_partial_class_gain(hybrid):
    classes = [
        {"name": n, "mean_f1": 0.5, "mean_mae_per_frame": 1.0}
        for n in (hybrid.ATTACHED, hybrid.FILAMENT)
    ]
    candidate = [dict(c) for c in classes]
    candidate[0]["mean_mae_per_frame"] = 0.8
    means = {
        "cosine30": {"count_guarded": {"mean_macro_f1": 0.5, "classes": classes}},
        "h_attached": {"count_guarded": {"mean_macro_f1": 0.5, "classes": candidate}},
    }
    result = hybrid.eligibility(means)["h_attached"]
    assert not result["eligible"]
    assert result["criteria"]["count_gain_attached_ciliates"]
    assert not result["criteria"]["count_gain_filamentous_bacteria"]


def test_deadline_hook_rejects_past(hybrid):
    from datetime import timedelta

    with pytest.raises(RuntimeError, match="Deadline"):
        hybrid.components.check_deadline(hybrid.night.now() - timedelta(seconds=1))


def test_report_renders_partial_no_metrics(hybrid):
    report = importlib.import_module("report_hybrid_night")
    summary = {
        "completed_jobs": 0,
        "means": {},
        "pooled_per_seed": {},
        "eligibility": {},
        "localization_diagnostics_per_seed": {},
    }
    text = report.render(
        summary, {"completed_components": 2, "costs": [], "cost_caveat": "partial"}
    )
    assert "0/6" in text and "2/30" in text and "NOT independent" in text


def test_worker_synthetic_end_to_end_freezes_before_outer(hybrid, config, monkeypatch, tmp_path):
    from datetime import timedelta

    monkeypatch.setattr(hybrid, "ROOT", tmp_path)
    monkeypatch.setattr(hybrid.old, "ROOT", tmp_path)
    monkeypatch.setattr(hybrid, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(hybrid, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(hybrid.night, "OUTPUTS", tmp_path / "locks")
    monkeypatch.setattr(hybrid, "DEADLINE", hybrid.night.now() + timedelta(hours=1))
    monkeypatch.setattr(hybrid, "verify", lambda _: None)
    names = config.classes.names
    subsets = {
        **hybrid.GROUPS,
        "general": names,
        "filamentous": (hybrid.FILAMENT,),
        "others": tuple(n for n in names if n != hybrid.FILAMENT),
    }
    configs, config_pins = {}, {}
    for source, subset in subsets.items():
        selected = tuple(
            t.model_copy(update={"label": i})
            for i, t in enumerate([t for t in config.classes.target if t.name in subset], 1)
        )
        cfg = config.model_copy(
            update={"classes": config.classes.model_copy(update={"target": selected})}
        )
        path = tmp_path / f"{source}.yaml"
        path.write_text(source, encoding="utf-8")
        configs[path] = cfg
        config_pins[source] = hybrid.old.pin(path)
    monkeypatch.setattr(hybrid, "load_config", lambda path: configs[path])
    sample_sets = {
        role: [
            Sample(
                f"{role}{i}",
                tmp_path / "unused",
                200,
                100,
                tuple((20 * j, 0, 20 * j + 10, 10) for j in range(7)),
                tuple(range(1, 8)),
            )
            for i in range(3)
        ]
        for role in ("fit", "cal", "outer")
    }

    def annotations(role):
        return pd.DataFrame(
            [
                {**pred(n, image_id=s.image_id, x=20 * j), "supervision": "target"}
                for s in sample_sets[role]
                for j, n in enumerate(names)
            ]
        )

    def predictions(source, role):
        return pd.DataFrame(
            [
                pred(n, image_id=s.image_id, x=20 * names.index(n))
                for s in sample_sets[role]
                for n in subsets[source]
            ]
        )[hybrid.components.PREDICTION_COLUMNS]

    manifest = pd.DataFrame(
        [
            {"image_id": s.image_id, "split": split, "group": s.image_id}
            for role, split in (("fit", "train"), ("cal", "validation"))
            for s in sample_sets[role]
        ]
    )
    monkeypatch.setattr(
        hybrid,
        "split_samples",
        lambda cfg, role: (
            sample_sets["fit" if role == "train" else "cal"],
            annotations("fit" if role == "train" else "cal"),
        ),
    )
    monkeypatch.setattr(hybrid, "load_split", lambda _: manifest)
    monkeypatch.setattr(
        hybrid.old, "outer_samples", lambda *_: (sample_sets["outer"], annotations("outer"))
    )
    job = {
        "fold": 0,
        "seed": 20261022,
        "run_id": "hybrids_s20261022",
        "recipe": "hybrids",
        "config": config_pins["general"],
        "components": {n: config_pins[n] for n in hybrid.GROUPS},
        "reuse": {},
    }
    for source in ("general", "filamentous", "others"):
        model = tmp_path / f"reused_{source}.pt"
        model.write_bytes(b"synthetic")
        reuse = {"config": config_pins[source], "model": hybrid.old.pin(model)}
        for role, key in (("cal", "calibration_predictions"), ("outer", "outer_predictions")):
            path = tmp_path / f"reused_{source}_{role}.parquet"
            predictions(source, role).to_parquet(path)
            reuse[key] = hybrid.old.pin(path)
        if source == "general":
            path = tmp_path / "general_calibration.json"
            hybrid.night.atomic_json(
                path,
                hybrid.old.calibrate(
                    config,
                    predictions(source, "cal"),
                    annotations("cal"),
                    manifest[manifest.split == "validation"],
                ),
            )
            reuse["calibration"] = hybrid.old.pin(path)
        job["reuse"][source] = reuse
    hybrid.night.atomic_json(hybrid.REPORTS / "registration.json", {"jobs": [job], "folds": [{}]})

    def fake_fit(cfg, samples, directory, deadline):
        assert {s.image_id for s in samples} == {s.image_id for s in sample_sets["fit"]}
        assert set(label for s in samples for label in s.labels) <= set(
            range(1, len(cfg.classes.names) + 1)
        )
        directory.mkdir(parents=True, exist_ok=True)
        for filename in ("model.pt", "resume_state.pt", "train_log.jsonl"):
            (directory / filename).write_bytes(b"synthetic")
        hybrid.night.atomic_json(
            directory / "complete.json", {"artifacts": [hybrid.old.pin(directory / "model.pt")]}
        )

    events = []

    def fake_predict(cfg, directory, samples, path):
        stage = "cal" if samples[0].image_id.startswith("cal") else "outer"
        if stage == "outer":
            assert (hybrid.job_dir(job, True) / "outer_authorization.json").is_file()
            assert all(
                (hybrid.job_dir(job, True) / f"calibration_{m}.json").is_file()
                for m in hybrid.REPLACEMENTS
            )
        events.append(stage)
        frame = predictions(directory.name, stage)
        frame.to_parquet(path)
        return frame

    monkeypatch.setattr(hybrid.components, "fit_detector", fake_fit)
    monkeypatch.setattr(hybrid.components, "detector_predictions", fake_predict)
    hybrid.worker(0)
    assert events == ["cal"] * 5 + ["outer"] * 5
    records = hybrid.verify_completed(job)
    assert len(records) == 12
    for record in records:
        assert record["policies"]["count_guarded"]["macro_f1"] == 1.0
    # Frozen calibration survives resumption without a different policy.
    hybrid.worker(0)
    assert len(hybrid.verify_completed(job)) == 12
