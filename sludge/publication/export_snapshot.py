"""Publish a byte-preserving, allowlisted research snapshot; never run experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEST = Path(__file__).resolve().parents[1]
MANIFEST = DEST / "publication/manifest.json"
TREES = (
    "configs", "src", "scripts", "tests", "model",
    "experiments_updated_2026_10_02", "outputs_updated_2026_10_02",
    "research/maa_transfer_v1", "research/sam_pilot_v1",
    "research/auto_micro_upstream/maa", "research/sam_upstream/segment_anything",
    "research/efficient_sam_upstream/efficient_sam",
)
ALLOWED = {".py", ".typed", ".json", ".jsonl", ".yaml", ".yml", ".md", ".csv", ".parquet", ".log", ".toml"}
SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_file(path):
    if path.stat().st_size > 20 * 1024**2:
        raise ValueError(f"Unexpected large publication artifact: {path}")
    if path.suffix != ".parquet":
        text = path.read_text(encoding="utf-8-sig")
        if SECRET.search(text):
            raise ValueError(f"Possible credential in {path.name}; not exported")
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)


def export(source):
    source = source.resolve(strict=True)
    if MANIFEST.exists():
        raise ValueError("Snapshot exists; refusing overwrite")
    candidates = [(p, p.name) for p in source.glob("*.md")]
    candidates += [(source / p, p) for p in ("pyproject.toml", "requirements.lock")]
    excluded = []
    for tree in TREES:
        for path in sorted((source / tree).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            if any(part in {"__pycache__", ".git", ".pytest_cache", ".ruff_cache"} for part in path.parts):
                continue
            if path.suffix not in ALLOWED:
                excluded.append({"path": relative, "bytes": path.stat().st_size,
                                 "reason": "not allowlisted: weights/images/locks/binary artifacts"})
                continue
            candidates.append((path, relative))
    for upstream in ("auto_micro_upstream", "sam_upstream", "efficient_sam_upstream"):
        for name in ("LICENSE", "README.md"):
            relative = f"research/{upstream}/{name}"
            candidates.append((source / relative, relative))
    copied = {}
    for path, relative in candidates:
        target_relative = "README_ORIGINAL.md" if relative == "README.md" else relative
        target = DEST / target_relative
        if target.exists():
            raise ValueError(f"Refuse overwrite: {target_relative}")
        inspect_file(path)
        original_hash = digest(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if digest(target) != original_hash:
            raise ValueError(f"Copy mismatch: {target_relative}")
        copied[target_relative] = relative
    files = []
    for path in sorted(DEST.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            inspect_file(path)
            relative = path.relative_to(DEST).as_posix()
            files.append({"path": relative, "bytes": path.stat().st_size,
                          "sha256": digest(path), "source_relative": copied.get(relative)})
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/PavelMarian/bio_collaboration",
        "source_base_commit": "48a8f6e01086af08efca78d3ebb17f4f760badb0",
        "source_dirty": True, "copied_files_are_byte_identical": True,
        "scope": list(TREES), "files": files, "excluded_within_scope": excluded,
        "excluded_entire_trees": ["data", "weights", ".venv", ".git", "caches"],
        "exclusions_note": "Original dataset images/annotations/archives and external sibling archives not enumerated or published. No model tensor files published. Weight hashes already present in original reports/registrations retained; excluded inventory sizes are not new weight integrity checks.",
    }
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"files": len(files), "bytes": sum(f["bytes"] for f in files),
                      "excluded": len(excluded), "extensions": dict(Counter(Path(f['path']).suffix for f in files))}))


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {record["path"] for record in manifest["files"]}
    actual = {p.relative_to(DEST).as_posix() for p in DEST.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    assert actual == expected | {"publication/manifest.json"}, "Unexpected/missing files"
    for record in manifest["files"]:
        path = DEST / record["path"]
        assert path.stat().st_size == record["bytes"], record["path"]
        assert digest(path) == record["sha256"], record["path"]
        inspect_file(path)
    print(f"Verified {len(expected)} files, SHA256/size/content checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--source", type=Path)
    options.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify() if args.verify else export(args.source)
