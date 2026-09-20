#!/usr/bin/env python3
"""Long-context lookback benchmark vs the authoritative full-file diarization.

Compares the speaker-attribution strategies on existing recordings:

  current   6 s fixed window, update every 4 s   (what production does today)
  12/4      12 s lookback, update every 4 s
  16/4      16 s lookback, update every 4 s
  20/5      20 s lookback, update every 5 s

Each snapshot is sent to ElevenLabs Scribe v2 batch (diarization) and folded
through the production LiveSessionStore (promotion rules, stable/mutable split).
Agreement is measured against each recording's own final full-file transcript
("final") using per-second temporal consensus; raw provider ids are never compared.

Structural output only: no transcript text, no raw responses. Private per-second
details land in the git-ignored results/private directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
for candidate in (SCRIPT_DIR.parent.parent / "backend", Path("/app")):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))
        break

from app.config import Settings  # noqa: E402
from app.services.live_sessions import LiveSessionStore  # noqa: E402
from app.services.providers.elevenlabs import transcribe_pcm_window  # noqa: E402
from app.services.transcription import NormalizedWord  # noqa: E402

import run_benchmark as rb  # noqa: E402

API = "http://localhost:8000"
PRIVATE = SCRIPT_DIR / "results" / "private"
GUARD_FILE = SCRIPT_DIR / "results" / "lookback-guard.json"
MAX_REQUESTS = 170

RECORDINGS = {
    "recent-2p": {
        "meeting_id": "0eb33b95c5874947ba19f1e196fe2fec",  # 88.8 s, human-verified final
        "final_meeting_id": "0eb33b95c5874947ba19f1e196fe2fec",
    },
    "multi-4p": {
        "meeting_id": "b1095740120b4b1e96db337da961ea63",  # 42.8 s
        "final_meeting_id": "diar-final-v2",
    },
}

CONFIGS = {
    "current_6_4": {"lookback": 6.0, "step": 4.0, "first_attempt": 6.0},
    "lookback_12_4": {"lookback": 12.0, "step": 4.0, "first_attempt": 8.0},
    "lookback_16_4": {"lookback": 16.0, "step": 4.0, "first_attempt": 8.0},
    "lookback_20_5": {"lookback": 20.0, "step": 5.0, "first_attempt": 10.0},
}


def read_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != 16_000 or wav_file.getnchannels() != 1:
            raise SystemExit("expected 16 kHz mono wav")
        frames = wav_file.getnframes()
        return wav_file.readframes(frames), frames / wav_file.getframerate()


def provider_intervals(words: list[NormalizedWord], offset: float, window: tuple[float, float]):
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


def snapshot_schedule(duration: float, config: dict) -> list[tuple[float, float]]:
    lookback, step, first = config["lookback"], config["step"], config["first_attempt"]
    schedule: list[tuple[float, float]] = []
    attempt = first
    while attempt <= duration + 1e-6:
        schedule.append((max(0.0, attempt - lookback), attempt))
        attempt += step
    return schedule


def final_turns(meeting_id: str) -> list[dict]:
    with httpx.Client(base_url=API, timeout=30.0) as client:
        payload = client.get(f"/api/v1/meetings/{meeting_id}/transcript").json()
    return payload["turns"]


def run_config(recording: dict, config_name: str, config: dict, pcm: bytes, duration: float,
               settings: Settings, speaker_count: int | None) -> dict:
    rb.GUARD_FILE = GUARD_FILE
    rb.MAX_REQUESTS = MAX_REQUESTS
    store = LiveSessionStore(ttl_seconds=3600)
    session = store.create(max_speakers=speaker_count)

    latencies: list[float] = []
    label_delays: list[float] = []
    uploaded = 0.0
    windows: list[dict] = []

    for sequence, (start, end) in enumerate(snapshot_schedule(duration, config), start=1):
        window_pcm = pcm[int(start * 32_000): int(end * 32_000)]
        guard = rb.load_guard()
        guard.check()
        started = time.perf_counter()
        words = transcribe_pcm_window(
            window_pcm, settings=settings, requested_speaker_count=speaker_count
        )
        latency = time.perf_counter() - started
        guard.record(label=f"{config_name}-w{sequence}", audio_seconds=end - start,
                     latency_seconds=latency, status=200)
        rb.save_guard(guard)
        latencies.append(latency)
        uploaded += end - start

        stable_until = max(start, end - config["step"])
        result = store.apply_window(
            session,
            provider_intervals=provider_intervals(words, start, (start, end)),
            window=(start, end),
            sequence=sequence,
            stable_until=stable_until,
        )
        for assignment in result["assignments"]:
            if not assignment["provisional"]:
                midpoint = (assignment["start"] + assignment["end"]) / 2
                label_delays.append(max(0.0, end + latency - midpoint))
        windows.append(
            {
                "sequence": sequence,
                "window": [start, end],
                "latency": round(latency, 3),
                "assignments": [
                    {
                        "label": item["canonical_speaker"],
                        "provisional": item["provisional"],
                        "start": item["start"],
                        "end": item["end"],
                    }
                    for item in result["assignments"]
                ],
                "candidates": result["candidate_speakers"],
                "ambiguous": result["ambiguous_speakers"],
            }
        )

    return {
        "config": config_name,
        "lookback": config["lookback"],
        "step": config["step"],
        "requests": len(windows),
        "uploaded_audio_seconds": round(uploaded, 1),
        "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else None,
        "median_label_delay_seconds": round(statistics.median(label_delays), 3)
        if label_delays
        else None,
        "p95_label_delay_seconds": round(
            sorted(label_delays)[int(len(label_delays) * 0.95)] if label_delays else 0.0, 3
        )
        if label_delays
        else None,
        "confirmed_speakers": session.confirmed_labels(),
        "confirmed_count": len(session.speakers),
        "candidate_speakers": len(session.candidates),
        "label_switches": session.label_switches,
        "ambiguous_segments": session.ambiguous_segments,
        "windows": windows,
    }


def consensus_seconds(config_result: dict, duration: float, resolution: float = 0.25) -> list[str | None]:
    """Per-instant label from the newest CONFIRMED assignment that covers it."""
    labels: list[str | None] = [None] * int(duration / resolution + 1)
    for window in config_result["windows"]:
        for assignment in window["assignments"]:
            if assignment["provisional"]:
                continue
            start_index = int(assignment["start"] / resolution)
            end_index = int(assignment["end"] / resolution) + 1
            for index in range(max(0, start_index), min(len(labels), end_index)):
                labels[index] = assignment["label"]
    return labels


def compare_to_final(config_result: dict, turns: list[dict], duration: float) -> dict:
    resolution = 0.25
    labels = consensus_seconds(config_result, duration, resolution)

    final_spans: dict[str, int] = {}
    attributed: dict[str, dict[str, int]] = {}
    unresolved = 0
    total = 0
    for index, label in enumerate(labels):
        time_at = index * resolution
        turn = next(
            (t for t in turns if t["start_seconds"] <= time_at <= t["end_seconds"]), None
        )
        if turn is None:
            continue
        speaker = turn["speaker"]
        final_spans[speaker] = final_spans.get(speaker, 0) + 1
        total += 1
        if label is None:
            unresolved += 1
        else:
            attributed.setdefault(speaker, {}).setdefault(label, 0)
            attributed[speaker][label] += 1

    # Optimal permutation: each final speaker matched to canonical labels greedily by
    # best share, one-to-one, so agreement is not understated by label naming.
    pairs: list[tuple[int, str, str]] = []
    for final_speaker, counts in attributed.items():
        for canonical, count in counts.items():
            pairs.append((count, final_speaker, canonical))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_final: set[str] = set()
    used_canonical: set[str] = set()
    matched = 0
    for count, final_speaker, canonical in pairs:
        if final_speaker in used_final or canonical in used_canonical:
            continue
        used_final.add(final_speaker)
        used_canonical.add(canonical)
        matched += count

    return {
        "agreement": round(matched / total, 3) if total else 0.0,
        "wrong_attribution": round((total - matched - unresolved) / total, 3) if total else 0.0,
        "unresolved": round(unresolved / total, 3) if total else 0.0,
        "final_speaker_seconds": {k: round(v * resolution, 1) for k, v in final_spans.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings", default=",".join(RECORDINGS))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--speaker-count", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    report: dict = {"results": {}}
    for recording_name in [name.strip() for name in args.recordings.split(",") if name.strip()]:
        recording = RECORDINGS[recording_name]
        audio = rb.REPO_ROOT / "data" / "meetings" / recording["meeting_id"] / "processing.wav"
        if not audio.is_file():
            raise SystemExit(f"audio missing: {audio}")
        pcm, duration = read_pcm(audio)
        turns = final_turns(recording["final_meeting_id"])
        final_speakers = sorted({turn["speaker"] for turn in turns})
        print(
            f"== {recording_name}: {duration:.1f}s final_speakers={final_speakers} turns={len(turns)}",
            flush=True,
        )
        report["results"][recording_name] = {"duration": duration,
                                             "final_speakers": final_speakers, "configs": {}}
        for config_name in [name.strip() for name in args.configs.split(",") if name.strip()]:
            result = run_config(recording, config_name, CONFIGS[config_name], pcm, duration,
                                settings, args.speaker_count)
            comparison = compare_to_final(result, turns, duration)
            summary = {
                "requests": result["requests"],
                "uploaded_audio_seconds": result["uploaded_audio_seconds"],
                "confirmed_speakers": result["confirmed_speakers"],
                "candidates": result["candidate_speakers"],
                "switches": result["label_switches"],
                "ambiguous_segments": result["ambiguous_segments"],
                "median_label_delay_seconds": result["median_label_delay_seconds"],
                "p95_label_delay_seconds": result["p95_label_delay_seconds"],
                "rolling_multiplier": round(result["uploaded_audio_seconds"] / duration, 2),
                **comparison,
            }
            print(json.dumps({config_name: summary}, ensure_ascii=False), flush=True)
            report["results"][recording_name]["configs"][config_name] = {
                **summary, "windows": result["windows"]
            }

    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "lookback-benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"private details: {PRIVATE / 'lookback-benchmark.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
