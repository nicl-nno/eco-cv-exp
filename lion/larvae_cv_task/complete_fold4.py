"""Complete the missing SE fold under the existing night protocol."""
from __future__ import annotations

import json
import traceback

from overnight import RUNS, now, run_one, write_json


def main() -> None:
    path = RUNS / "overnight_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    key = "resnet34_unet_se/fold_4"
    if key in status["runs"]:
        print(f"Already complete: {key}")
        return
    status["state"] = "running"
    try:
        run_one("resnet34_unet_se", 4, status)
    except Exception as exc:
        status["errors"].append({"run": key, "error": str(exc), "time": now()})
        traceback.print_exc()
        raise
    finally:
        status["active"] = None
        status["state"] = "finished"
        status["updated"] = now()
        write_json(path, status)


if __name__ == "__main__":
    main()
