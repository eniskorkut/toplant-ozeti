#!/usr/bin/env python3
"""Comprehensive calibration and policy study for conservative live speaker attribution.

Fetches and caches ElevenLabs window diarization for the accepted 12s/4s schedule
on recent-2p and multi-4p recordings, collects utterance-level and timeline-level
confidence statistics, and evaluates policies P1..P10 against final full-file Scribe v2.
"""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
import wave
from pathlib import Path

from app.config import Settings
from app.services.live_sessions import LiveSessionStore
from app.services.providers.elevenlabs import transcribe_pcm_window
from app.services.transcription import NormalizedWord

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PRIVATE = SCRIPT_DIR / "results" / "private"
CACHE_FILE = PRIVATE / "window-cache.json"

LOOKBACK = 12.0
STEP = 4.0
FIRST_ATTEMPT = 8.0
RESOLUTION = 0.25

RECORDINGS = {
    "recent-2p": {
        "meeting_id": "0eb33b95c5874947ba19f1e196fe2fec",
        "final_meeting_id": "0eb33b95c5874947ba19f1e196fe2fec",
    },
    "multi-4p": {
        "meeting_id": "b1095740120b4b1e96db337da961ea63",
        "final_meeting_id": "diar-final-v2",
    },
}


def read_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.getnframes()
        return wav_file.readframes(frames), frames / wav_file.getframerate()


def snapshot_schedule(duration: float) -> list[tuple[float, float]]:
    schedule: list[tuple[float, float]] = []
    attempt = FIRST_ATTEMPT
    while attempt <= duration + 1e-6:
        schedule.append((max(0.0, attempt - LOOKBACK), attempt))
        attempt += STEP
    return schedule


def get_cached_windows(settings: Settings) -> dict[str, list[dict]]:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, list[dict]] = {}
    if CACHE_FILE.is_file():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    for name, spec in RECORDINGS.items():
        audio_path = REPO_ROOT / "data" / "meetings" / spec["meeting_id"] / "processing.wav"
        pcm, duration = read_pcm(audio_path)
        sched = snapshot_schedule(duration)
        rec_cached = cache.setdefault(name, [])
        cached_seqs = {w["sequence"] for w in rec_cached}

        for seq, (start, end) in enumerate(sched, start=1):
            if seq in cached_seqs:
                continue
            window_pcm = pcm[int(start * 32_000) : int(end * 32_000)]
            print(f"[{name}] Transcribing window {seq}/{len(sched)}: [{start:.1f} - {end:.1f}]...", flush=True)
            t0 = time.perf_counter()
            words = transcribe_pcm_window(window_pcm, settings=settings)
            latency = time.perf_counter() - t0
            rec_cached.append(
                {
                    "sequence": seq,
                    "start": start,
                    "end": end,
                    "latency": round(latency, 3),
                    "words": [
                        {
                            "speaker_id": w.speaker_id,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                        }
                        for w in words
                    ],
                }
            )
            # Save cache incrementally
            CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        rec_cached.sort(key=lambda w: w["sequence"])

    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def words_to_provider_intervals(
    word_dicts: list[dict], offset: float, window: tuple[float, float]
) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    for w in word_dicts:
        spk = w.get("speaker_id")
        if not spk:
            continue
        start = max(window[0], offset + w["start"])
        end = min(window[1], offset + w["end"])
        if end <= start:
            continue
        intervals.setdefault(spk, []).append((start, end))
    return intervals


def get_final_turns(meeting_id: str) -> list[dict]:
    url = f"http://localhost:8000/api/v1/meetings/{meeting_id}/transcript"
    req = urllib.request.urlopen(url)
    return json.loads(req.read().decode("utf-8"))["turns"]


def main():
    settings = Settings(_env_file=REPO_ROOT / ".env.local")
    cache = get_cached_windows(settings)
    print("All windows cached successfully!")
    for name in RECORDINGS:
        print(f"  {name}: {len(cache[name])} windows")


if __name__ == "__main__":
    main()
