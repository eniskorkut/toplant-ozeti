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

# Languages are declared by the orchestration layer, never inferred inside the
# low-level STT runner: VoxConverse is English, the private meeting is Turkish.
VOXCONVERSE_LANGUAGE = "en"
REAL_MEETING_LANGUAGE = "tr"
TURKISH_CONTROLLED = [
    {"id": "sample_far", "meeting_id": "fcdf1737901d41ddacd14429d39a4183", "language": "tr"},
    {"id": "sample_near", "meeting_id": "798b2efc586542d780eef3dce5dbbc8b", "language": "tr"},
]
MODES = ("heuristic", "heuristic_nfa", "dtw")


def extraction_plan() -> list[dict]:
    """Every extraction job with its explicitly assigned language."""
    plan = [
        {"id": file_id, "language": VOXCONVERSE_LANGUAGE, "kind": "voxconverse"}
        for file_id in VOXCONVERSE_FILES
    ]
    plan.append({"id": REAL_MEETING_ID, "language": REAL_MEETING_LANGUAGE, "kind": "real"})
    plan.extend(
        {
            "id": item["id"],
            "language": item["language"],
            "kind": "turkish_controlled",
            "meeting_id": item["meeting_id"],
        }
        for item in TURKISH_CONTROLLED
    )
    return plan


def extract(
    file_id: str,
    audio_container: str,
    mounts: list[str],
    language: str,
    *,
    force: bool,
) -> None:
    for mode in MODES:
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
            "--language", language,
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
    parser.add_argument("--force", action="store_true", help="ignore cached outputs")
    args = parser.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for job in extraction_plan():
        if job["kind"] == "voxconverse":
            print(f"VoxConverse {job['id']} (language={job['language']})")
            extract(
                job["id"],
                f"/audio/{job['id']}.wav",
                ["-v", f"{VOXCONVERSE_AUDIO}:/audio:ro"],
                job["language"],
                force=args.force,
            )
        elif job["kind"] == "real":
            print(f"real four-speaker meeting (language={job['language']})")
            extract(
                job["id"],
                f"/audio/{job['id']}/processing.wav",
                ["-v", f"{MEETINGS_DIR}:/audio:ro"],
                job["language"],
                force=args.force,
            )
        else:
            print(f"controlled Turkish recording {job['id']} (language={job['language']})")
            extract(
                job["id"],
                f"/audio/{job['meeting_id']}/processing.wav",
                ["-v", f"{MEETINGS_DIR}:/audio:ro"],
                job["language"],
                force=args.force,
            )
    print(f"cached outputs in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
