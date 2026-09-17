#!/usr/bin/env python3
"""Parallel range downloader for the official VoxConverse 0.3 dev audio archive.

Provenance (recorded, not inferred):
    dataset   : VoxConverse (dev set audio)
    source    : https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
    version   : 0.3 annotations (github.com/joonson/voxconverse master, commit recorded
                in benchmarks/diarization/datasets/voxconverse/)
    license   : CC BY 4.0, research purposes

The archive is git-ignored and stays under benchmarks/diarization/datasets/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

URL = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"
DIAR_DIR = Path(__file__).resolve().parent
DESTINATION = DIAR_DIR / "datasets" / "voxconverse" / "voxconverse_dev_wav.zip"
PROVENANCE = DIAR_DIR / "datasets" / "voxconverse" / "provenance.json"
WORKERS = 8


def curl(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["curl", "-fsSL", *args],
        capture_output=capture,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )


def remote_size() -> int:
    headers = curl("-I", URL, capture=True).stdout
    for line in headers.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    raise SystemExit("could not determine remote size")


def fetch_range(start: int, end: int, target: Path) -> None:
    curl("-r", f"{start}-{end}", "-o", str(target), URL)


def verify_zip(path: Path) -> None:
    """Full ZIP integrity test. Aborts the download if the archive is corrupt."""
    try:
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise SystemExit(f"ZIP integrity test failed (corrupt member: {bad_member})")
            members = len(archive.namelist())
    except zipfile.BadZipFile as exc:
        raise SystemExit(f"ZIP integrity test failed (not a valid archive): {exc}") from exc
    print(f"zip integrity OK ({members} members)")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size()

    if DESTINATION.exists() and DESTINATION.stat().st_size == total and not args.force:
        print(f"archive already complete ({total} bytes)")
    else:
        chunk_size = math.ceil(total / WORKERS)
        parts: list[Path] = []
        ranges = []
        for index in range(WORKERS):
            start = index * chunk_size
            end = min(start + chunk_size - 1, total - 1)
            if start > end:
                break
            part = DESTINATION.with_suffix(f".part{index}")
            parts.append(part)
            ranges.append((start, end, part))

        print(f"downloading {total} bytes in {len(ranges)} parallel ranges")
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(fetch_range, start, end, part) for start, end, part in ranges]
            for future in futures:
                future.result()

        with DESTINATION.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        output.write(chunk)
                part.unlink()

        size = DESTINATION.stat().st_size
        if size != total:
            raise SystemExit(f"size mismatch: expected {total}, got {size}")
        print(f"downloaded {size} bytes")

    verify_zip(DESTINATION)

    PROVENANCE.write_text(
        json.dumps(
            {
                "dataset": "VoxConverse dev set audio",
                "version": "0.3 annotations (github.com/joonson/voxconverse master)",
                "source_url": URL,
                "license": "CC BY 4.0 (research purposes)",
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "size_bytes": total,
                "sha256": sha256_of(DESTINATION),
                "zip_integrity": "ok",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {PROVENANCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
