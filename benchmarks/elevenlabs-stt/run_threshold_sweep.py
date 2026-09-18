#!/usr/bin/env python3
"""Four-request diarization threshold sweep on the SAME existing meeting audio.

Budget: exactly the thresholds listed below, one request each, no retries. Automatic
speaker count only (no num_speakers), so diarization_threshold is valid. Raw responses
and private transcripts stay in the git-ignored results/ directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import wave
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO_ROOT = HARNESS.parent.parent
RESULTS = HARNESS / "results"
RAW = RESULTS / "raw"
PRIVATE = RESULTS / "private"

sys.path.insert(0, str(HARNESS))
from el_parse import parse_words, response_metrics  # noqa: E402
from run_benchmark import API_URL, COMMON_FIELDS, api_key, audio_seconds  # noqa: E402

MEETING_ID = "b1095740120b4b1e96db337da961ea63"
AUDIO = REPO_ROOT / "data" / "meetings" / MEETING_ID / "processing.wav"
THRESHOLDS = [0.22, 0.18, 0.14, 0.10]
MAX_REQUESTS = 4


def request_threshold(key: str, threshold: float) -> dict:
    fields = {
        **COMMON_FIELDS,
        "language_code": "tur",
        "diarize": "true",
        "diarization_threshold": str(threshold),
    }
    stdin_config = (
        f'header = "xi-api-key: {key}"\n'
        'header = "Accept: application/json"\n'
        'request = "POST"\n'
        f'url = "{API_URL}"\n'
        'silent\nshow-error\n'
    )
    target = RAW / f"threshold-{threshold:.2f}.json"
    command = ["curl", "-K", "-", "-o", str(target), "-w", "%{http_code}"]
    for name, value in fields.items():
        command += ["-F", f"{name}={value}"]
    command += ["-F", f"file=@{AUDIO}"]

    started = time.perf_counter()
    completed = subprocess.run(
        command, input=stdin_config, capture_output=True, text=True, check=False
    )
    latency = time.perf_counter() - started
    status = int((completed.stdout or "0").strip() or 0)
    return {"threshold": threshold, "status": status, "latency_seconds": round(latency, 3)}


def write_private_transcript(threshold: float, payload: dict) -> int:
    words = parse_words(payload)
    labels: dict[str, str] = {}
    turns: list[list] = []
    for word in words:
        label = (
            labels.setdefault(word.speaker, f"Kişi {len(labels) + 1}")
            if word.speaker
            else "Bilinmeyen"
        )
        if turns and turns[-1][0] == label:
            turns[-1][1] = f"{turns[-1][1]} {word.text}"
            turns[-1][3] = word.end
        else:
            turns.append([label, word.text, word.start, word.end])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / f"threshold-{threshold:.2f}.txt").write_text(
        "\n".join(f"[{entry[2]:8.3f}] {entry[0]}: {entry[1]}" for entry in turns) + "\n",
        encoding="utf-8",
    )
    return len(turns)


def main() -> int:
    key = api_key()
    RAW.mkdir(parents=True, exist_ok=True)
    if not AUDIO.is_file():
        raise SystemExit(f"source audio missing: {AUDIO}")
    seconds = audio_seconds(AUDIO)
    print(f"source audio: {seconds} s | planned additional upload: {seconds * len(THRESHOLDS):.2f} s")

    summary = {
        "meeting_id": MEETING_ID,
        "audio_seconds": seconds,
        "requests_planned": MAX_REQUESTS,
        "thresholds": {},
    }
    for index, threshold in enumerate(THRESHOLDS, start=1):
        print(f"[{index}/{MAX_REQUESTS}] threshold {threshold:.2f}", flush=True)
        result = request_threshold(key, threshold)
        if result["status"] == 429:
            print("quota/rate limit reached - stopping immediately")
            break
        if result["status"] != 200:
            print(f"request failed with HTTP {result['status']} - stopping (no retries)")
            break
        payload = json.loads((RAW / f"threshold-{threshold:.2f}.json").read_text(encoding="utf-8"))
        metrics = response_metrics(payload)
        turns = write_private_transcript(threshold, payload)
        summary["thresholds"][f"{threshold:.2f}"] = {
            "latency_seconds": result["latency_seconds"],
            "turns_written": turns,
            **metrics,
        }
        print(
            f"    speakers={metrics['distinct_speakers']} words={metrics['words']} "
            f"tagged={metrics['speaker_tagged_words']} untagged={metrics['untagged_words']} "
            f"turns={metrics['speaker_turns']} flips={metrics['rapid_flips']} "
            f"latency={result['latency_seconds']}s",
            flush=True,
        )

    (RESULTS / "threshold-sweep.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {RESULTS / 'threshold-sweep.json'} (safe metrics) and private transcripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
