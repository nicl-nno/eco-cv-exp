"""Bounded recovery of transient public-data failures; no network access in tests."""

import importlib.util
import urllib.error
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def reconstruction(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location(
        "release_reconstruction_test", folder / "reconstruct_release_split.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        module.night, "now", lambda: datetime.fromisoformat("2026-10-21T03:00:00+03:00")
    )
    return module


def test_transient_http_failure_is_retried(reconstruction, monkeypatch, tmp_path):
    fetch = Mock(
        side_effect=[urllib.error.HTTPError("https://example.test", 500, "error", {}, None), None]
    )
    monkeypatch.setattr(reconstruction.download, "fetch", fetch)
    reconstruction.fetch_with_retry("archive.zip", tmp_path / "archive.zip", "sha")
    assert fetch.call_count == 2


def test_network_retries_are_bounded(reconstruction, monkeypatch, tmp_path):
    fetch = Mock(side_effect=urllib.error.URLError("temporary network error"))
    monkeypatch.setattr(reconstruction.download, "fetch", fetch)
    with pytest.raises(urllib.error.URLError):
        reconstruction.fetch_with_retry("archive.zip", tmp_path / "archive.zip", "sha")
    assert fetch.call_count == 3


@pytest.mark.parametrize(
    "error",
    [
        ValueError("digest mismatch"),
        urllib.error.HTTPError("https://example.test", 403, "denied", {}, None),
    ],
)
def test_digest_or_permission_failure_is_not_retried(reconstruction, monkeypatch, tmp_path, error):
    fetch = Mock(side_effect=error)
    monkeypatch.setattr(reconstruction.download, "fetch", fetch)
    with pytest.raises(type(error)):
        reconstruction.fetch_with_retry("archive.zip", tmp_path / "archive.zip", "sha")
    assert fetch.call_count == 1
