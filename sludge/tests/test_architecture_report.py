"""CPU report helpers: seed SD, undefined precision and complete metric coverage."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("report_architecture_night")


def test_seed_statistics_and_undefined_values(report):
    assert report.stats([1.0, 3.0]) == {"mean": 2.0, "sd": pytest.approx(2**0.5)}
    assert report.stats([None, 0.0]) == {"mean": None, "sd": None}
    assert report.number(None) == "NA"
    assert report.mean_sd(0.123456, 0.009876) == "0.1235 ± 0.0099"


def test_markdown_table_preserves_zero_and_columns(report):
    rendered = report.table(["A", "B"], [[0, "NA"], [1, "value"]])
    assert rendered.splitlines() == ["| A | B |", "| --- | --- |", "| 0 | NA |", "| 1 | value |"]


def test_existing_summary_completeness_and_criteria(report):
    path = report.run.REPORTS / "summary.json"
    if not path.exists():
        pytest.skip("Night artifact unavailable")
    summary = report.read_json(path)
    assert summary["completed_pipeline_jobs"] == 12
    assert len(summary["pooled_per_seed"]) == 8
    assert report.run.eligibility(summary["means"]) == summary["eligibility"]
    for method in report.run.METHODS:
        for policy in report.run.POLICIES:
            mean = summary["means"][method][policy]
            values = [
                summary["pooled_per_seed"][f"{method}_s{s}"][policy]["macro_f1"]
                for s in report.run.SEEDS
            ]
            assert mean["mean_macro_f1"] == pytest.approx(report.stats(values)["mean"])
            assert mean["std_macro_f1"] == pytest.approx(report.stats(values)["sd"])
            assert len(mean["classes"]) == 7
