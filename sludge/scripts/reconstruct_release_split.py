"""Recover old-source split metadata in isolation; no training or model evaluation.

Root Yandex archive digests were read on 2026-10-21. Matching summary counts alone
does NOT verify historical membership. Only an exact original manifest digest does.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

import fetch_updated_archives as download
import overnight_experiments as night

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_updated_2026_10_02/release_reconstruction"
REPORTS = ROOT / "experiments_updated_2026_10_02/release_reconstruction"
RAW = ROOT / "data/raw/release_reconstruction"
EXPECTED_MANIFEST = "998d1089a1cdb1738691c61ae2b57f73d286993d983299d0dd6770fa67dd1551"
ARCHIVES = {
    "Student_1.zip": "81ff39db7284a6593490ee8d2d69970b4b9bb88c0f9a9cc986b8931e2f1dd5c7",
    "Student_2.zip": "f5a64809d8da43d489678fba5087438603ac7bb0d42231a55e43c2829afd3190",
    "Student_4.zip": "06f56fca58644638896057d679f191b64fe1d6f7856f6906ac2f7f0c874c5eeb",
    "Student_6.zip": "7c618cb00545d850effb30a1cb47aaf968ce1e07d32d75f530125ce5b19ffa6f",
    "Student_7.zip": "4008e1bd36601c814ea4352bc293dbbb466fdc6fd58bac95a8fd02ab7b4546a3",
    "Student_8.zip": "63325c5e34bfe2eae3dc789f2cccef4f68bca1e77643eb75aaacd43ebc2d8dc6",
    "Student_9.zip": "723b5dcb06ebca4a258044c7387c3223a1d96e5b10c9fea869945ba5c478a493",
    "Student_10.zip": "d3e01bf1323d494267cf15f4611a2a050f4de5e67f4cb53d73da495817a5b669",
    "Student_11 CHECKED.zip": "20ef460ef1877484151c77e91bb4988ef82b26eae7f1796a93d7233e0fa47433",
    "31.01.24.zip": "66d9d31f3ccdb866d63de1ca6b777748deccc10c4e6f5b6eae8aa77694cbe557",
}


def stage(name: str) -> None:
    print(name, flush=True)
    night.atomic_json(
        OUT / "state.json",
        {
            "stage": name,
            "status": "completed" if name == "completed" else "running",
            "pid": os.getpid(),
            "updated_at": night.now().isoformat(),
            "deadline": night.DEADLINE.isoformat(),
        },
    )
    if night.now() >= night.DEADLINE:
        raise RuntimeError("Night deadline reached")


def fetch_with_retry(name: str, target: Path, expected: str) -> None:
    """Retry transient server/network failures at most three times; never a bad digest."""
    for attempt in range(3):
        if night.now() >= night.DEADLINE:
            raise RuntimeError("Night deadline reached")
        try:
            download.fetch(name, target, expected)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {
                429,
                500,
                502,
                503,
                504,
            }:
                raise
            if attempt == 2:
                raise
            print(f"Transient download failure for {name}; retry {attempt + 1}/2", flush=True)
            time.sleep(2 ** (attempt + 1))


def fetch_archives() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    download.FOLDER = "/"
    # Reuse a previously verified original Student_6 without touching its source copy.
    existing = ROOT / "data/raw/размечено FINAL/Student_6.zip"
    target = RAW / "Student_6.zip"
    if (
        not target.exists()
        and existing.exists()
        and download.digest(existing) == ARCHIVES[target.name]
    ):
        temporary = target.with_suffix(".zip.part")
        shutil.copy2(existing, temporary)
        os.replace(temporary, target)
    for name, expected in ARCHIVES.items():
        stage(f"download:{name}")
        fetch_with_retry(name, RAW / name, expected)
    night.atomic_json(
        REPORTS / "archive_provenance.json",
        {
            "public_key": download.PUBLIC_KEY,
            "folder": "/",
            "pinned_on": "2026-10-21",
            "archives": ARCHIVES,
            "historical_release_archive_digests_available": False,
            "warning": "Current root files may differ from files used at historical training time.",
        },
    )


def compare() -> None:
    from sludge_micro.config import load_config
    from sludge_micro.prepare import load_prepared
    from sludge_micro.reporting import read_json
    from sludge_micro.runs import split_ids
    from sludge_micro.split import load_split

    old_config = load_config(ROOT / "configs/release_reconstruction.yaml")
    current_config = load_config(ROOT / "configs/updated_2026_10_02.yaml")
    old = load_split(old_config)
    current = load_split(current_config)
    lock = read_json(REPORTS / "split_lock.json")
    passport = read_json(ROOT / "model/passport.json")
    actual = {
        "train_images_after_filter": len(split_ids(old_config, "train")),
        "validation_images": int((old["split"] == "validation").sum()),
        "test_images": int((old["split"] == "test").sum()),
    }
    expected = {key: passport["training"]["split"][key] for key in actual}
    old_training = old[old["image_id"].isin(split_ids(old_config, "train"))]
    validation = current[current["split"] == "validation"]
    overlaps = validation[validation["pixel_sha256"].isin(old_training["pixel_sha256"])]
    pairs = overlaps[["image_id", "pixel_sha256"]].merge(
        old_training[["image_id", "pixel_sha256"]],
        on="pixel_sha256",
        suffixes=("_current", "_reconstructed"),
    )
    _, annotations = load_prepared(old_config)
    supports = {}
    for split in ("validation", "test"):
        ids = set(old.loc[old["split"] == split, "image_id"])
        target = annotations[
            annotations["image_id"].isin(ids) & (annotations["supervision"] == "target")
        ]
        supports[split] = {
            "reconstructed": {
                c: int((target["class_name"] == c).sum()) for c in old_config.classes.names
            },
            "passport": {c["name"]: c["support"] for c in passport["evaluation"][split]["classes"]},
        }
    verified = lock["manifest_sha256"] == EXPECTED_MANIFEST
    night.atomic_json(
        REPORTS / "reconstruction_comparison.json",
        {
            "expected_manifest_sha256": EXPECTED_MANIFEST,
            "actual_manifest_sha256": lock["manifest_sha256"],
            "exact_manifest_match": verified,
            "historical_membership_verified": verified,
            "image_counts": {"expected": expected, "actual": actual, "match": expected == actual},
            "class_supports": supports,
            "current_validation_images": len(validation),
            "current_validation_pixel_overlap_with_reconstructed_training": len(overlaps),
            "overlap_pairs": pairs.to_dict(orient="records"),
            "warning": "No model evaluated. Unless manifest matches exactly, overlap is with a "
            "RECONSTRUCTION, not proven release training. Counts alone cannot verify membership.",
        },
    )


def work() -> None:
    from sludge_micro.audit import run_audit
    from sludge_micro.config import load_config
    from sludge_micro.prepare import run_prepare
    from sludge_micro.runtime import apply_runtime
    from sludge_micro.split import run_split
    from sludge_micro.variants import run_variants

    config = load_config(ROOT / "configs/release_reconstruction.yaml")
    apply_runtime(config.runtime)
    fetch_archives()
    steps = [
        ("audit", REPORTS / "data_audit.json", run_audit),
        ("prepare", REPORTS / "prepare_report.json", run_prepare),
        (
            "variants",
            config.resolve(config.paths.processed_dir) / "variants/prefer_student_6/images.parquet",
            run_variants,
        ),
        ("split", REPORTS / "split_lock.json", run_split),
    ]
    for name, marker, action in steps:
        stage(name)
        if not marker.exists():
            action(config)
    stage("compare")
    compare()
    stage("completed")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with night.FileLock(OUT / "reconstruction.lock"):
        if night.now() >= night.DEADLINE:
            raise RuntimeError("Night deadline reached")
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1", "CUDA_VISIBLE_DEVICES": "-1"},
        )
        try:
            code = process.wait(timeout=max(1, (night.DEADLINE - night.now()).total_seconds()))
        except subprocess.TimeoutExpired:
            night.stop_worker(process)
            night.atomic_json(OUT / "state.json", {"stage": "deadline_stopped"})
            raise SystemExit(2) from None
        if code:
            state_path = OUT / "state.json"
            state = (
                json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            )
            night.atomic_json(state_path, {**state, "status": "failed", "exit_code": code})
            raise SystemExit(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    work() if args.worker else main()
