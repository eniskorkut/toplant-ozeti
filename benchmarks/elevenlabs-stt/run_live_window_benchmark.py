#!/usr/bin/env python3
"""Rolling-window configuration benchmark on the known 22 s two-person sample.

Compares two candidate configurations for the live diarization loop:

  A: 4 s window / 1 s overlap (step 3)
  B: 6 s window / 2 s overlap (step 4)

Each window is sent to ElevenLabs batch Scribe v2 (diarize, pcm_s16le_16) and
folded through the production matching code (LiveSessionStore) so the measured
stability reflects the real implementation. Raw responses are never written;
only safe aggregates are printed. Private results go to the git-ignored
results/private directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
for candidate in (SCRIPT_DIR.parent.parent / "backend", Path("/app")):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from app.config import Settings  # noqa: E402
from app.services.live_sessions import LiveSessionStore  # noqa: E402
from app.services.providers.elevenlabs import transcribe_pcm_window  # noqa: E402

import run_benchmark as rb  # noqa: E402

MEETING_ID = "8266cc18930b40e6a3740c48bc543705"
AUDIO = rb.REPO_ROOT / "data" / "meetings" / MEETING_ID / "processing.wav"
PRIVATE = SCRIPT_DIR / "results" / "private"
GUARD_FILE = SCRIPT_DIR / "results" / "live-window-guard.json"
MAX_REQUESTS = 12

CONFIGS = {
    "A_4s_1s": {"window": 4.0, "overlap": 1.0},
    "B_6s_2s": {"window": 6.0, "overlap": 2.0},
}


def read_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != 16_000 or wav_file.getnchannels() != 1:
            raise SystemExit("expected 16 kHz mono wav")
        if wav_file.getsampwidth() != 2:
            raise SystemExit("expected 16-bit samples")
        frames = wav_file.getnframes()
        return wav_file.readframes(frames), frames / wav_file.getframerate()


def window_starts(duration: float, window: float, overlap: float) -> list[float]:
    step = window - overlap
    starts: list[float] = []
    start = 0.0
    while start + window <= duration + 1e-6:
        starts.append(round(start, 3))
        start += step
    return starts


def provider_intervals(words, *, offset: float, window: tuple[float, float]):
    intervals: dict[str, list[tuple[float, float]]] = {}
    for word in words:
        if word.speaker_id is None:
            continue
        start = max(window[0], offset + word.start)
        end = min(window[1], offset + word.end)
        if end <= start:
            continue
        intervals.setdefault(word.speaker_id, []).append((start, end))
    return intervals


def run_config(label: str, config: dict, pcm: bytes, duration: float, settings: Settings) -> dict:
    rb.GUARD_FILE = GUARD_FILE
    rb.MAX_REQUESTS = MAX_REQUESTS
    store = LiveSessionStore(ttl_seconds=3600)
    session = store.create()

    starts = window_starts(duration, config["window"], config["overlap"])
    latencies: list[float] = []
    assignments_per_window: list[dict] = []
    label_delays: list[float] = []

    for sequence, start in enumerate(starts, start=1):
        end = start + config["window"]
        start_sample = int(round(start * 16_000))
        end_sample = int(round(end * 16_000))
        window_pcm = pcm[start_sample * 2 : end_sample * 2]

        guard = rb.load_guard()
        guard.check()
        started = time.perf_counter()
        words = transcribe_pcm_window(
            window_pcm,
            settings=settings,
            requested_speaker_count=None,
        )
        latency = time.perf_counter() - started
        guard.record(
            label=f"{label}-w{sequence}",
            audio_seconds=config["window"],
            latency_seconds=latency,
            status=200,
        )
        rb.save_guard(guard)
        latencies.append(latency)

        intervals = provider_intervals(words, offset=start, window=(start, end))
        result = store.apply_window(
            session, provider_intervals=intervals, window=(start, end), sequence=sequence
        )
        assignments_per_window.append(
            {
                "window": [start, end],
                "assignments": result["assignments"],
                "new_speakers": result["new_speakers"],
            }
        )
        for assignment in result["assignments"]:
            # Label for speech around the window midpoint arrives when this window
            # returns (window ends + provider latency).
            midpoint = (assignment["start"] + assignment["end"]) / 2
            label_delays.append(max(0.0, end + latency - midpoint))

    canonical = sorted(session.speakers)
    summary = {
        "config": label,
        "window_seconds": config["window"],
        "overlap_seconds": config["overlap"],
        "requests": len(starts),
        "uploaded_audio_seconds": round(len(starts) * config["window"], 3),
        "detected_canonical_speakers": len(canonical),
        "canonical_speakers": canonical,
        "label_switch_events": session.label_switches,
        "median_provider_latency_seconds": round(statistics.median(latencies), 3),
        "median_speaker_label_delay_seconds": round(statistics.median(label_delays), 3)
        if label_delays
        else None,
        "assignments_per_window": assignments_per_window,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), action="append")
    parser.add_argument("--check-key", action="store_true")
    args = parser.parse_args()

    if args.check_key:
        key = rb.api_key()
        print(f"key_present={bool(key)}")
        return 0

    if not AUDIO.is_file():
        raise SystemExit(f"sample audio missing: {AUDIO}")
    labels = args.config or sorted(CONFIGS)
    pcm, duration = read_pcm(AUDIO)
    settings = Settings()
    print(f"sample: {MEETING_ID} duration={duration:.2f}s windows={labels}", flush=True)

    summaries = []
    for label in labels:
        summary = run_config(label, CONFIGS[label], pcm, duration, settings)
        summaries.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "assignments_per_window"}), flush=True)

    PRIVATE.mkdir(parents=True, exist_ok=True)
    target = PRIVATE / "live-window-benchmark.json"
    target.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"private details: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
