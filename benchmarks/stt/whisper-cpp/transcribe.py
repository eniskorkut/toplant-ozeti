"""Benchmark-only whisper.cpp transcription. Emits one JSON object on stdout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

BINARY = "/usr/local/bin/whisper-cli"
VERSION_FILE = Path("/opt/whisper.cpp/VERSION")

DURATION_RE = re.compile(r"\((\d+) samples, ([\d.]+) sec\)")
TOTAL_TIME_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")
LOAD_TIME_RE = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")


def parse_engine_seconds(stderr: str) -> float | None:
    match = TOTAL_TIME_RE.search(stderr)
    return round(float(match.group(1)) / 1000, 3) if match else None


def parse_load_seconds(stderr: str) -> float | None:
    match = LOAD_TIME_RE.search(stderr)
    return round(float(match.group(1)) / 1000, 3) if match else None


def parse_audio_duration(stderr: str) -> float | None:
    match = DURATION_RE.search(stderr)
    return float(match.group(2)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="/models/ggml-small-q5_1.bin")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    command = [
        BINARY,
        "-m",
        args.model,
        "-f",
        args.audio,
        "-l",
        args.language,
        "-t",
        str(args.threads),
        "-bs",
        str(args.beam_size),
        "-nt",
        "-ng",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)

    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "engine": "whisper.cpp",
                    "error": f"whisper-cli exited with code {completed.returncode}",
                    "stderr_tail": "\n".join(completed.stderr.strip().splitlines()[-10:]),
                },
                ensure_ascii=False,
            )
        )
        return 1

    transcript = " ".join(line.strip() for line in completed.stdout.splitlines() if line.strip())
    engine_seconds = parse_engine_seconds(completed.stderr)

    print(
        json.dumps(
            {
                "engine": "whisper.cpp",
                "engine_version": VERSION_FILE.read_text(encoding="utf-8").strip()
                if VERSION_FILE.exists()
                else "unknown",
                "model": Path(args.model).name,
                "compute_type": "ggml-quantized",
                "threads": args.threads,
                "beam_size": args.beam_size,
                "vad": False,
                "language": args.language,
                "audio_duration_seconds": parse_audio_duration(completed.stderr),
                "engine_reported_seconds": engine_seconds,
                "model_load_seconds": parse_load_seconds(completed.stderr),
                "inference_seconds": engine_seconds,
                "transcript": transcript,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
