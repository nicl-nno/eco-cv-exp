"""Fetch the October 2 microorganism annotations from the public Yandex folder.

Run from any directory: python scripts/fetch_updated_archives.py. Existing files
are verified before reuse; downloads are atomic and checked against the pinned
SHA-256 values from the public resource listing on 2026-10-21.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

PUBLIC_KEY = "https://disk.yandex.ru/d/Al2bRZQ4F8sDFw"
FOLDER = "/Updated 02.10.2026/"
API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
DESTINATION = Path(__file__).resolve().parents[1] / "data/raw/updated_2026_10_02"
# Source archive -> name expected by the existing COCO reader, pinned digest.
ARCHIVES = {
    "1.zip": ("Student_1.zip", "953df3d421fd6fcf3b164a5f32f62adfae3cf668017591024772ba1988b0262f"),
    "2.zip": ("Student_2.zip", "a2bb1d6dba9eaec74f86119dcac53a51b8f0a40a5fe242ab4fc2bd100e3d0383"),
    "4.zip": ("Student_4.zip", "541e6367c7e533779dcf14c14437a9aad6ee03a63a7cbc7651393a9f7ca92e66"),
    "6.zip": ("Student_6.zip", "c675c3278c017a8b1e72bd3dfb90b4d5dd8faa12d5047246c5d6ec6dc291396f"),
    "7.zip": ("Student_7.zip", "d0b307659288869c6c31d93227a7b6cc629ccde47c5df7f6d5ea2d5d610a10c9"),
    "8.zip": ("Student_8.zip", "179f724f6366f749663d9882d10843778ec4ea14fff906d24370c20828967bb2"),
    "9.zip": ("Student_9.zip", "24f4bba66de7a3d57d71e8dff24079f05b448f07b1f1d82af1c7af358a25d0e9"),
    "10.zip": (
        "Student_10.zip",
        "11032a172ba4bc8bcac69220ffc34733c182c6169d5a92c40d2c5643c330dcad",
    ),
    "11.zip": (
        "Student_11 CHECKED.zip",
        "8a3a01422c221d81a905008e80e5a5e2b91c3a8afcab28eb6166ab42edbb797c",
    ),
    "31.01.24.zip": (
        "31.01.24.zip",
        "3d8f6a658dcb862c4dcf9ea56eb7451e89263ead8c5c96cfe968337189d1ba17",
    ),
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fetch(name: str, target: Path, expected: str) -> None:
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Existing file has a different digest: {target}")
        print(f"verified {target.name}")
        return
    query = urllib.parse.urlencode({"public_key": PUBLIC_KEY, "path": FOLDER + name})
    with urllib.request.urlopen(f"{API}?{query}", timeout=45) as response:
        url = json.load(response)["href"]
    temporary = target.with_suffix(".zip.part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        if digest(temporary) != expected:
            raise ValueError(f"Downloaded archive changed or is incomplete: {name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"downloaded {target.name}")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, (filename, expected) in ARCHIVES.items():
        fetch(name, DESTINATION / filename, expected)


if __name__ == "__main__":
    main()
