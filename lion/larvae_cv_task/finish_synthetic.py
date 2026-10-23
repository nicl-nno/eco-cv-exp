"""Finish the fold-0 synthetic comparison after its training process completes."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
FOLD = HERE / "runs" / "copy_paste" / "fold_0"
STATUS = FOLD / "followup_status.json"


def status(**fields: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(fields, indent=2), encoding="utf-8")


def main() -> None:
    try:
        deadline = time.monotonic() + 4 * 3600
        while not (FOLD / "result.json").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("training did not write result.json within four hours")
            time.sleep(20)
        status(state="selecting")
        subprocess.run([sys.executable, "select_checkpoint.py", "--config", "config_synthetic.yaml",
                        "--fold-dir", str(FOLD)], cwd=HERE, check=True)
        status(state="evaluating")
        subprocess.run([sys.executable, "evaluate_tiling.py", "--config", "config_synthetic.yaml",
                        "--checkpoint", str(FOLD / "selected_full_miou.pt"),
                        "--stride", "448", "--tta", "flip4",
                        "--output", str(FOLD / "tiling_comparison.json")], cwd=HERE, check=True)
        status(state="done", result=str(FOLD / "tiling_comparison.json"))
    except Exception as exc:
        status(state="error", error=str(exc))
        raise


if __name__ == "__main__":
    main()
