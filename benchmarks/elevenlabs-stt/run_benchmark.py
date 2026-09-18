#!/usr/bin/env python3
"""ElevenLabs Scribe v2 benchmark client (host-side, standard library + curl).

- Reads the key from the ELEVENLABS_API_KEY environment variable only.
- The key is passed to curl through a config on stdin, never via argv and never
  written to disk.
- No automatic retries: one requested benchmark call = one network call.
- Raw responses and private transcripts stay in the git-ignored results/ directory.

Usage:
    python3 benchmarks/elevenlabs-stt/run_benchmark.py --check-key
    python3 benchmarks/elevenlabs-stt/run_benchmark.py --call real-known4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from el_parse import (  # noqa: E402
    ElevenConfigurationError,
    ElevenQuotaError,
    ElevenRequestGuard,
    group_turns,
    parse_words,
    redact_error,
    response_metrics,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEETINGS = REPO_ROOT / "data" / "meetings"
VOXCONVERSE = REPO_ROOT / "benchmarks" / "diarization" / "datasets" / "voxconverse"
RESULTS = Path(__file__).resolve().parent / "results"
RAW = RESULTS / "raw"
PRIVATE = RESULTS / "private"
GUARD_FILE = RESULTS / "request-guard.json"

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
MAX_REQUESTS = 6
# Git-ignored, harness-local env file: an alternative to exporting the variable.
ENV_LOCAL = Path(__file__).resolve().parent / ".env.local"

REAL_MEETING = "b1095740120b4b1e96db337da961ea63"
TURKISH_FAR = "fcdf1737901d41ddacd14429d39a4183"
TURKISH_NEAR = "798b2efc586542d780eef3dce5dbbc8b"

CALLS = {
    "real-known4": {
        "audio": MEETINGS / REAL_MEETING / "processing.wav",
        "fields": {"language_code": "tur", "diarize": "true", "num_speakers": "4"},
    },
    "real-auto": {
        "audio": MEETINGS / REAL_MEETING / "processing.wav",
        "fields": {"language_code": "tur", "diarize": "true"},
    },
    "turkish-far": {
        "audio": MEETINGS / TURKISH_FAR / "processing.wav",
        "fields": {"language_code": "tur", "diarize": "false"},
    },
    "turkish-near": {
        "audio": MEETINGS / TURKISH_NEAR / "processing.wav",
        "fields": {"language_code": "tur", "diarize": "false"},
    },
    "voxconverse-whmpa": {
        "audio": VOXCONVERSE / "audio" / "whmpa.wav",
        "fields": {"language_code": "eng", "diarize": "true", "num_speakers": "2"},
    },
    "voxconverse-wjhgf": {
        "audio": VOXCONVERSE / "audio" / "wjhgf.wav",
        "fields": {"language_code": "eng", "diarize": "true", "num_speakers": "5"},
    },
}

COMMON_FIELDS = {
    "model_id": "scribe_v2",
    "timestamps_granularity": "word",
    "tag_audio_events": "false",
    "no_verbatim": "false",
}


def log(message: str) -> None:
    print(message, flush=True)


def read_env_local(path: Path = ENV_LOCAL) -> dict[str, str]:
    """Minimal KEY=VALUE reader for the git-ignored harness-local env file."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def api_key() -> str:
    """Prefer the process environment; fall back to the git-ignored .env.local.

    The value is never logged, printed or written back to disk.
    """
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        key = read_env_local().get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise ElevenConfigurationError(
            "ELEVENLABS_API_KEY is not configured (environment or "
            "benchmarks/elevenlabs-stt/.env.local)"
        )
    return key


def audio_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return round(wav_file.getnframes() / wav_file.getframerate(), 3)


def load_guard() -> ElevenRequestGuard:
    guard = ElevenRequestGuard(MAX_REQUESTS)
    if GUARD_FILE.exists():
        state = json.loads(GUARD_FILE.read_text(encoding="utf-8"))
        guard.used = int(state.get("requests", 0))
        guard.audio_seconds_sent = float(state.get("total_audio_seconds_sent", 0.0))
        guard.calls = state.get("calls", [])
    return guard


def save_guard(guard: ElevenRequestGuard) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    GUARD_FILE.write_text(json.dumps(guard.summary(), indent=2), encoding="utf-8")


def request(label: str, key: str) -> dict:
    spec = CALLS[label]
    audio = spec["audio"]
    if not audio.is_file():
        raise SystemExit(f"audio missing for {label}")
    if not RESULTS.exists():
        RESULTS.mkdir(parents=True)
    RAW.mkdir(parents=True, exist_ok=True)

    fields = {**COMMON_FIELDS, **spec["fields"]}
    # curl config is read from stdin: the key never appears in argv or in a file.
    stdin_config = (
        f'header = "xi-api-key: {key}"\n'
        'header = "Accept: application/json"\n'
        'request = "POST"\n'
        f'url = "{API_URL}"\n'
        'silent\nshow-error\n'
    )
    command = ["curl", "-K", "-", "-o", str(RAW / f"{label}.json"), "-w", "%{http_code}"]
    for name, value in fields.items():
        command += ["-F", f"{name}={value}"]
    command += ["-F", f"file=@{audio}"]

    started = time.perf_counter()
    completed = subprocess.run(
        command, input=stdin_config, capture_output=True, text=True, check=False
    )
    latency = time.perf_counter() - started
    status_text = (completed.stdout or "").strip()
    try:
        status = int(status_text) if status_text else 0
    except ValueError:
        status = 0

    guard = load_guard()
    guard.check()
    guard.record(label=label, audio_seconds=audio_seconds(audio), latency_seconds=latency, status=status)
    save_guard(guard)

    if status == 401:
        raise SystemExit(redact_error("ElevenLabs rejected the API key (HTTP 401)", [key]))
    if status == 429 or "quota" in (completed.stderr or "").lower():
        raise ElevenQuotaError(
            redact_error(f"quota/plan problem (HTTP {status}) - stopping further calls", [key])
        )
    if status != 200:
        raise SystemExit(
            redact_error(
                f"request failed (HTTP {status}): {(completed.stderr or '')[:200]}", [key]
            )
        )
    return json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))


def write_private_transcript(label: str, payload: dict) -> int:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    turns = group_turns(parse_words(payload))
    lines = [f"[{turn.start:8.3f}] {turn.speaker}: {turn.text}" for turn in turns]
    (PRIVATE / f"{label}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(turns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-key", action="store_true")
    parser.add_argument("--call", choices=sorted(CALLS))
    args = parser.parse_args()

    if args.check_key:
        try:
            api_key()
            log("elevenlabs_key_configured = true")
        except ElevenConfigurationError:
            log("elevenlabs_key_configured = false")
            return 1
        return 0

    if not args.call:
        parser.error("--call is required unless --check-key is used")
    key = api_key()
    payload = request(args.call, key)
    metrics = response_metrics(payload)
    turns = write_private_transcript(args.call, payload)
    log(f"{args.call}: {json.dumps({**metrics, 'turns_written': turns}, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
