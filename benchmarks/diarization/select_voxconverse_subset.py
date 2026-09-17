#!/usr/bin/env python3
"""Deterministic VoxConverse 0.3 dev subset selection.

Reads the reference RTTM annotations first, then selects recordings without any
knowledge of model performance:

    4 recordings with exactly 2 speakers
    4 recordings with exactly 3 speakers
    4 recordings with >= 4 speakers

Rules: skip files shorter than 30 s, prefer files <= 10 minutes, sort by
duration ascending with the file id as tie breaker, take the first 4 valid ones.

WAV durations are read from the audio archive headers (no full extraction).
"""

from __future__ import annotations

import argparse
import json
import struct
import zipfile
from pathlib import Path

DIAR_DIR = Path(__file__).resolve().parent
DATASET_DIR = DIAR_DIR / "datasets" / "voxconverse"
RTTM_DIR = DATASET_DIR / "voxconverse" / "dev"
ARCHIVE = DATASET_DIR / "voxconverse_dev_wav.zip"
SUBSET_FILE = DIAR_DIR / "voxconverse_subset.txt"
METADATA_FILE = DATASET_DIR / "subset-metadata.json"

MIN_DURATION_SECONDS = 30.0
PREFERRED_MAX_SECONDS = 600.0
PER_GROUP = 4


def parse_rttm(path: Path) -> dict:
    speakers: set[str] = set()
    speech_seconds = 0.0
    overlap_seconds = 0.0
    segments: list[tuple[float, float, str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        speakers.add(speaker)
        segments.append((start, start + duration, speaker))

    intervals = sorted((start, end) for start, end, _ in segments)

    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    union_seconds = sum(end - start for start, end in merged)

    boundaries = sorted({point for start, end in intervals for point in (start, end)})
    overlap_seconds = 0.0
    for index in range(len(boundaries) - 1):
        left, right = boundaries[index], boundaries[index + 1]
        active = sum(1 for start, end in intervals if start <= left and end >= right)
        if active >= 2:
            overlap_seconds += right - left

    return {
        "num_speakers": len(speakers),
        "segments": len(segments),
        "summed_speech_seconds": round(sum(end - start for start, end in intervals), 3),
        "union_speech_seconds": round(union_seconds, 3),
        "overlap_seconds": round(overlap_seconds, 3),
    }


def wav_duration_from_zip(archive: zipfile.ZipFile, member: str) -> float | None:
    """Read the WAV header inside the archive without extracting the whole file.

    VoxConverse dev WAVs are 16 kHz mono PCM, but the header is parsed properly:
    RIFF chunks are scanned and `data` size / byte rate is used.
    """
    with archive.open(member) as handle:
        header = handle.read(4096)

    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return None

    offset = 12
    byte_rate = None
    while offset + 8 <= len(header):
        chunk_id = header[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", header, offset + 4)[0]
        if chunk_id == b"fmt " and offset + 8 + 16 <= len(header):
            byte_rate = struct.unpack_from("<I", header, offset + 8 + 8)[0]
        if chunk_id == b"data":
            if byte_rate:
                return chunk_size / byte_rate
            return None
        offset += 8 + chunk_size + (chunk_size % 2)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if SUBSET_FILE.exists() and not args.force:
        print(f"{SUBSET_FILE} already exists; use --force to recompute")
        return 0

    if not RTTM_DIR.is_dir():
        raise SystemExit(f"RTTM directory missing: {RTTM_DIR}")
    if not ARCHIVE.exists():
        raise SystemExit(f"audio archive missing: {ARCHIVE}")

    provenance_path = ARCHIVE.parent / "provenance.json"
    if not provenance_path.exists():
        raise SystemExit(
            f"provenance missing: {provenance_path} — run download_voxconverse.py first"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("zip_integrity") != "ok":
        raise SystemExit("refusing to select: archive did not pass the ZIP integrity test")
    if provenance.get("size_bytes") != ARCHIVE.stat().st_size:
        raise SystemExit("refusing to select: archive size differs from the verified download")

    candidates: list[dict] = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = {Path(name).stem: name for name in archive.namelist() if name.endswith(".wav")}

        for rttm_path in sorted(RTTM_DIR.glob("*.rttm")):
            file_id = rttm_path.stem
            member = members.get(file_id)
            if member is None:
                continue
            duration = wav_duration_from_zip(archive, member)
            if duration is None or duration < MIN_DURATION_SECONDS:
                continue
            reference = parse_rttm(rttm_path)
            if reference["num_speakers"] < 2:
                continue
            candidates.append(
                {
                    "file_id": file_id,
                    "member": member,
                    "duration_seconds": round(duration, 3),
                    "reference": reference,
                }
            )

    def group_of(count: int) -> str:
        if count == 2:
            return "two"
        if count == 3:
            return "three"
        return "four_plus"

    selected: list[dict] = []
    groups = {"two": [], "three": [], "four_plus": []}
    for candidate in candidates:
        groups[group_of(candidate["reference"]["num_speakers"])].append(candidate)

    for name, entries in groups.items():
        entries.sort(key=lambda item: (item["duration_seconds"], item["file_id"]))
        preferred = [entry for entry in entries if entry["duration_seconds"] <= PREFERRED_MAX_SECONDS]
        pool = preferred if len(preferred) >= PER_GROUP else entries
        chosen = pool[:PER_GROUP]
        for entry in chosen:
            entry["group"] = name
        selected.extend(chosen)
        print(f"{name}: {len(entries)} candidates, selected {len(chosen)}")
        for entry in chosen:
            print(
                f"  {entry['file_id']:<8} {entry['duration_seconds']:8.2f} s  "
                f"speakers={entry['reference']['num_speakers']} "
                f"speech={entry['reference']['union_speech_seconds']:.1f} s "
                f"overlap={entry['reference']['overlap_seconds']:.1f} s"
            )

    SUBSET_FILE.write_text(
        "\n".join(entry["file_id"] for entry in selected) + "\n", encoding="utf-8"
    )
    METADATA_FILE.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(f"wrote {SUBSET_FILE} and {METADATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
