"""Bounded CPU regression tests, independent of scientific model experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import overnight_experiments as night

OUTPUT = night.ROOT / "outputs_updated_2026_10_02/verification_2026_10_21"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with night.FileLock(OUTPUT / "verification.lock"):
        state_path = OUTPUT / "slow_tests_state.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if previous["status"] in {"completed", "failed"}:
                raise RuntimeError("Prior verification exists; inspect it instead of overwriting")
        if night.now() >= night.DEADLINE:
            raise RuntimeError("Night deadline reached")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "slow",
            "--maxfail=1",
            f"--junitxml={OUTPUT / 'slow_tests.xml'}",
        ]
        state = {
            "status": "running",
            "supervisor_pid": os.getpid(),
            "started_at": night.now().isoformat(),
            "deadline": night.DEADLINE.isoformat(),
            "command": command,
            "data": "synthetic test fixtures only",
            "device": "cpu",
        }
        with (OUTPUT / "slow_tests.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=night.ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "-1"},
            )
            state["worker_pid"] = process.pid
            night.atomic_json(state_path, state)
            try:
                code = process.wait(timeout=max(1, (night.DEADLINE - night.now()).total_seconds()))
                state.update(status="completed" if code == 0 else "failed", exit_code=code)
            except subprocess.TimeoutExpired:
                night.stop_worker(process)
                state["status"] = "deadline_stopped"
            finally:
                state["finished_at"] = night.now().isoformat()
                night.atomic_json(state_path, state)


if __name__ == "__main__":
    main()
