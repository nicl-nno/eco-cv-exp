"""Operational checks for the bounded night queue; no real training is launched."""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest


@pytest.fixture
def queue_module(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/overnight_experiments.py"
    spec = importlib.util.spec_from_file_location("overnight_experiments_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(module, "OUTPUTS", tmp_path / "outputs")
    return module


def test_registered_queue_has_twelve_training_jobs_and_is_immutable(queue_module):
    queue = queue_module.make_queue()
    assert len(queue) == 13
    assert queue[0]["run_id"] == "release_baseline"
    assert len({r["run_id"] for r in queue}) == 13
    assert queue_module.make_queue() == queue
    path = queue_module.REPORTS / "queue_configs" / f"{queue[1]['run_id']}.yaml"
    path.write_text("train: {epochs: 99}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to change"):
        queue_module.make_queue()


def test_deadline_prevents_any_worker_start(queue_module, monkeypatch):
    monkeypatch.setattr(
        queue_module, "DEADLINE", datetime.fromisoformat("2000-01-01T08:00:00+03:00")
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Worker started after deadline")

    monkeypatch.setattr(queue_module.subprocess, "Popen", forbidden)
    queue_module.run_queue()
    state = json.loads((queue_module.OUTPUTS / "queue_state.json").read_text())
    assert state["status"] == "deadline_reached"
    assert all(q["status"] == "pending" for q in state["queue"])


def test_process_lock_refuses_duplicate_and_releases_on_close(queue_module, tmp_path):
    lock = tmp_path / "test.lock"
    code = (
        "import runpy,sys; from pathlib import Path; "
        "m=runpy.run_path(sys.argv[1]); "
        "guard=m['FileLock'](Path(sys.argv[2])); guard.__enter__(); guard.__exit__()"
    )
    command = [sys.executable, "-c", code, queue_module.__file__, str(lock)]
    with queue_module.FileLock(lock):
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert result.returncode != 0
        assert "Another overnight process" in result.stderr
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_deadline_stop_covers_windows_virtualenv_children(queue_module, monkeypatch):
    from unittest.mock import Mock

    process = Mock(pid=12345)
    process.poll.return_value = None
    launch = Mock()
    monkeypatch.setattr(queue_module.os, "name", "nt")
    monkeypatch.setattr(queue_module.subprocess, "run", launch)
    queue_module.stop_worker(process)
    launch.assert_called_once_with(
        ["taskkill", "/PID", "12345", "/T", "/F"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    process.wait.assert_called_once_with(timeout=30)


def test_stop_does_not_target_an_already_exited_process(queue_module, monkeypatch):
    from unittest.mock import Mock

    process = Mock()
    process.poll.return_value = 0
    launch = Mock()
    monkeypatch.setattr(queue_module.subprocess, "run", launch)
    queue_module.stop_worker(process)
    launch.assert_not_called()
    process.terminate.assert_not_called()


def test_queue_refuses_restart_while_orphan_worker_is_alive(queue_module):
    with (
        queue_module.FileLock(queue_module.OUTPUTS / "gpu.lock"),
        pytest.raises(RuntimeError, match="Another overnight process"),
    ):
        queue_module.run_queue()
    assert not (queue_module.OUTPUTS / "queue_state.json").exists()
