#!/usr/bin/env python3
"""Host-side STT extraction for the overlap-merge benchmark (cached).

Runs whisper.cpp (heuristic and DTW timestamp modes) for the 12 VoxConverse subset
files and the real four-speaker meeting. Raw outputs are cached under results/raw/
(git-ignored); nothing here touches the runtime database or production settings.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DIAR_DIR = REPO_ROOT / "benchmarks" / "diarization"
VOXCONVERSE_AUDIO = DIAR_DIR / "datasets" / "voxconverse" / "audio"
MEETINGS_DIR = REPO_ROOT / "data" / "meetings"
WHISPER_IMAGE = "mi-bench-whisper-cpp:local"
WHISPER_MODEL_DIR = REPO_ROOT / "benchmarks" / "stt" / ".cache" / "whisper-cpp"
WHISPER_MODEL = "/models/ggml-large-v3-turbo-q8_0.bin"

RAW_DIR = Path(__file__).resolve().parent / "results" / "raw"

VOXCONVERSE_FILES = [
    "qpylu", "fxgvy", "szsyz", "rtvuw", "gwtwd", "bwzyf",
    "whmpa", "bkwns", "syiwe", "jiqvr", "jyirt", "wjhgf",
]
REAL_MEETING_ID = "b1095740120b4b1e96db337da961ea63"


def extract(file_id: str, audio_host: Path, audio_container: str, mounts: list[str], *, force: bool) -> None:
    for mode in ("heuristic", "heuristic_nfa", "dtw"):
        target = RAW_DIR / f"{file_id}--{mode}.json"
        if target.exists() and not force:
            continue
        command = [
            "docker", "run", "--rm",
            *mounts,
            "-v", f"{Path(__file__).resolve().parent}:/bench:ro",
            "-v", f"{RAW_DIR}:/out",
            "-v", f"{WHISPER_MODEL_DIR}:/models:ro",
            WHISPER_IMAGE,
            "python", "/bench/stt_words.py",
            "--audio", audio_container,
            "--model", WHISPER_MODEL,
            "--language", "tr",
            "--timestamps", mode,
            "--out", f"/out/{file_id}--{mode}.json",
        ]
        print(f"  {file_id} [{mode}]", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
        if completed.returncode != 0:
            raise SystemExit(f"STT failed for {file_id} [{mode}]:\n{completed.stderr[-1500:]}")
        print(f"    {completed.stdout.strip()}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("VoxConverse subset (12 files)")
    for file_id in VOXCONVERSE_FILES:
        extract(
            file_id,
            VOXCONVERSE_AUDIO / f"{file_id}.wav",
            f"/audio/{file_id}.wav",
            ["-v", f"{VOXCONVERSE_AUDIO}:/audio:ro"],
            force=args.force,
        )

    print("real four-speaker meeting")
    extract(
        REAL_MEETING_ID,
        MEETINGS_DIR / REAL_MEETING_ID / "processing.wav",
        f"/audio/{REAL_MEETING_ID}/processing.wav",
        ["-v", f"{MEETINGS_DIR}:/audio:ro"],
        force=args.force,
    )
    print(f"cached outputs in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
