#!/usr/bin/env python3
"""Live speaker-diarization diagnostics + rolling-window benchmark.

Structural data only: word counts, request-local speaker counts, latency and
canonical mapping statistics. Transcript text is never printed or stored.

Configs (Phase 3): A 4/1, B 6/2, C 8/2, D 10/3. The same windows are folded
through the production LiveSessionStore so measurements reflect the shipped
matching code (margin gating, gap/fragment merges, request-local ids).

Usage (backend container):

    python /tmp/bench/run_live_diarization_diag.py \
        [--configs A_4s_1s,B_6s_2s,C_8s_2s,D_10s_3s] [--final-meeting-id elevenlabs-thr022]
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
DEFAULT_AUDIO_MEETING = "b1095740120b4b1e96db337da961ea63"  # 4 physical speakers
PRIVATE = SCRIPT_DIR / "results" / "private"
GUARD_FILE = SCRIPT_DIR / "results" / "diarization-guard.json"
MAX_REQUESTS = 40

CONFIGS = {
    "A_4s_1s": {"window": 4.0, "overlap": 1.0},
    "B_6s_2s": {"window": 6.0, "overlap": 2.0},
    "C_8s_2s": {"window": 8.0, "overlap": 2.0},
    "D_10s_3s": {"window": 10.0, "overlap": 3.0},
}


def read_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != 16_000 or wav_file.getnchannels() != 1:
            raise SystemExit("expected 16 kHz mono wav")
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


def window_structure(words: list[NormalizedWord], offset: float) -> dict:
    """Provider-side structure of one response (anonymous ids only)."""
    per_speaker: dict[str, int] = {}
    untagged = 0
    intervals: dict[str, list[tuple[float, float]]] = {}
    transitions = 0
    previous: str | None = None
    for word in words:
        if word.speaker_id is None:
            untagged += 1
            previous = None
            continue
        per_speaker[word.speaker_id] = per_speaker.get(word.speaker_id, 0) + 1
        intervals.setdefault(word.speaker_id, []).append(
            (offset + word.start, offset + word.end)
        )
        if previous is not None and word.speaker_id != previous:
            transitions += 1
        previous = word.speaker_id

    overlapping_pairs = 0
    speaker_names = sorted(intervals)
    for index, left in enumerate(speaker_names):
        for right in speaker_names[index + 1 :]:
            for left_span in intervals[left]:
                for right_span in intervals[right]:
                    if min(left_span[1], right_span[1]) > max(left_span[0], right_span[0]):
                        overlapping_pairs += 1

    return {
        "words": len(words),
        "provider_speakers": len(per_speaker),
        "speaker_word_counts": [per_speaker[name] for name in speaker_names],
        "tagged_words": sum(per_speaker.values()),
        "untagged_words": untagged,
        "speaker_transitions": transitions,
        "overlapping_word_pairs": overlapping_pairs,
        "intervals": intervals,
    }


def run_config(
    label: str,
    config: dict,
    pcm: bytes,
    duration: float,
    settings: Settings,
    speaker_count: int | None = None,
) -> dict:
    rb.GUARD_FILE = GUARD_FILE
    rb.MAX_REQUESTS = MAX_REQUESTS
    store = LiveSessionStore(ttl_seconds=3600)
    session = store.create()

    starts = window_starts(duration, config["window"], config["overlap"])
    latencies: list[float] = []
    label_delays: list[float] = []
    windows: list[dict] = []
    total_provider_observations = 0
    multi_speaker_windows = 0
    single_speaker_windows = 0
    empty_windows = 0
    evidence_counts: dict[str, int] = {}
    pending_assignments = 0

    for sequence, start in enumerate(starts, start=1):
        end = start + config["window"]
        window_pcm = pcm[int(start * 32_000) : int(end * 32_000)]

        guard = rb.load_guard()
        guard.check()
        started = time.perf_counter()
        words = transcribe_pcm_window(
            window_pcm, settings=settings, requested_speaker_count=speaker_count
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

        structure = window_structure(words, start)
        total_provider_observations += structure["provider_speakers"]
        if structure["provider_speakers"] == 0:
            empty_windows += 1
        elif structure["provider_speakers"] == 1:
            single_speaker_windows += 1
        else:
            multi_speaker_windows += 1

        result = store.apply_window(
            session,
            provider_intervals=structure["intervals"],
            window=(start, end),
            sequence=sequence,
        )
        pending_assignments += result["ambiguous_speakers"]
        for assignment in result["assignments"]:
            evidence_counts[assignment["evidence"]] = (
                evidence_counts.get(assignment["evidence"], 0) + 1
            )
            midpoint = (assignment["start"] + assignment["end"]) / 2
            label_delays.append(max(0.0, end + latency - midpoint))

        windows.append(
            {
                "sequence": sequence,
                "window": [start, end],
                "duration": config["window"],
                "latency_seconds": round(latency, 3),
                "words": structure["words"],
                "provider_speakers": structure["provider_speakers"],
                "speaker_word_counts": structure["speaker_word_counts"],
                "provider_intervals": [
                    [[round(start, 3), round(end, 3)] for start, end in spans]
                    for spans in structure["intervals"].values()
                ],
                "tagged_words": structure["tagged_words"],
                "untagged_words": structure["untagged_words"],
                "speaker_transitions": structure["speaker_transitions"],
                "overlapping_word_pairs": structure["overlapping_word_pairs"],
                "assignments": [
                    {
                        "canonical_speaker": item["canonical_speaker"],
                        "is_new": item["is_new"],
                        "confidence": item["confidence"],
                        "evidence": item["evidence"],
                        "start": item["start"],
                        "end": item["end"],
                    }
                    for item in result["assignments"]
                ],
                "new_speakers": result["new_speakers"],
                "ambiguous_speakers": result["ambiguous_speakers"],
            }
        )

    canonical = sorted(session.speakers)
    summary = {
        "config": label,
        "window_seconds": config["window"],
        "overlap_seconds": config["overlap"],
        "requests": len(starts),
        "uploaded_audio_seconds": round(len(starts) * config["window"], 3),
        "median_latency_seconds": round(statistics.median(latencies), 3),
        "median_speaker_label_delay_seconds": round(statistics.median(label_delays), 3)
        if label_delays
        else None,
        "total_provider_speaker_observations": total_provider_observations,
        "canonical_speaker_count": len(canonical),
        "canonical_speakers": canonical,
        "label_switch_events": session.label_switches,
        "fragmentation_events": session.label_switches,
        "merge_assignments": evidence_counts.get("merged-fragment", 0),
        "gap_assignments": evidence_counts.get("gap", 0),
        "overlap_assignments": evidence_counts.get("overlap", 0),
        "windows_with_multiple_provider_speakers": multi_speaker_windows,
        "windows_with_single_provider_speaker": single_speaker_windows,
        "windows_without_words": empty_windows,
        "ambiguous_provider_speakers": pending_assignments,
        "pending_seconds": round(
            sum(max(0.0, end - start) for start, end in session.pending_intervals), 3
        ),
        "windows": windows,
    }
    return summary


def _overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def final_comparison(meeting_id: str, rolling_summary: dict) -> dict:
    """Rolling canonical timelines vs the final full-file Scribe diarization."""
    with httpx.Client(base_url=API, timeout=30.0) as client:
        transcript = client.get(f"/api/v1/meetings/{meeting_id}/transcript").json()

    final_spans: dict[str, list[tuple[float, float]]] = {}
    for turn in transcript["turns"]:
        final_spans.setdefault(turn["speaker"], []).append(
            (float(turn["start_seconds"]), float(turn["end_seconds"]))
        )
    rolling_spans: dict[str, list[tuple[float, float]]] = {}
    for window in rolling_summary["windows"]:
        for assignment in window["assignments"]:
            rolling_spans.setdefault(assignment["canonical_speaker"], []).append(
                (float(assignment["start"]), float(assignment["end"]))
            )

    def total(spans: dict[str, list[tuple[float, float]]]) -> float:
        return sum(
            end - start for values in spans.values() for start, end in values
        )

    final_total = total(final_spans) or 1.0
    agreement: dict[str, dict] = {}
    best_seconds_total = 0.0
    collisions: dict[str, list[str]] = {}
    fragments = 0
    for final_speaker, spans in sorted(final_spans.items()):
        speaker_total = sum(end - start for start, end in spans) or 1.0
        overlaps = {
            canonical: sum(
                _overlap(final_span, rolling_span)
                for final_span in spans
                for rolling_span in rolling
            )
            for canonical, rolling in rolling_spans.items()
        }
        best = max(overlaps.items(), key=lambda item: (item[1], item[0]), default=(None, 0.0))
        best_seconds_total += best[1]
        if best[0] is not None:
            collisions.setdefault(best[0], []).append(final_speaker)
        if best[1] / speaker_total < 0.5:
            fragments += 1
        agreement[final_speaker] = {
            "best_canonical": best[0],
            "agreement": round(best[1] / speaker_total, 3),
            "overlap_seconds": round(best[1], 2),
        }

    unmatched_rolling = [
        canonical
        for canonical in rolling_spans
        if all(canonical != entry["best_canonical"] for entry in agreement.values())
    ]
    return {
        "final_meeting_id": meeting_id,
        "final_speakers": sorted(final_spans),
        "rolling_canonical_speakers": sorted(rolling_spans),
        "speaker_count_delta": len(final_spans) - len(rolling_spans),
        "final_speech_seconds": round(final_total, 2),
        "rolling_speech_seconds": round(total(rolling_spans), 2),
        "time_overlap_agreement": round(best_seconds_total / final_total, 3),
        "per_final_speaker": agreement,
        "collisions": {k: v for k, v in collisions.items() if len(v) > 1},
        "final_speakers_belOW_50pct_agreement": fragments,
        "rolling_speakers_unmatched": unmatched_rolling,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-meeting", default=DEFAULT_AUDIO_MEETING)
    parser.add_argument("--configs", default=",".join(sorted(CONFIGS)))
    parser.add_argument("--tag", default=None, help="label suffix for private outputs")
    parser.add_argument("--final-meeting-id", default=None)
    parser.add_argument("--speaker-count", type=int, default=None)
    args = parser.parse_args()

    audio = rb.REPO_ROOT / "data" / "meetings" / args.audio_meeting / "processing.wav"
    if not audio.is_file():
        raise SystemExit(f"audio missing: {audio}")
    pcm, duration = read_pcm(audio)
    settings = Settings()
    labels = [label.strip() for label in args.configs.split(",") if label.strip()]
    print(f"audio={args.audio_meeting} duration={duration:.2f}s configs={labels}", flush=True)

    summaries = []
    for label in labels:
        summary = run_config(
            label, CONFIGS[label], pcm, duration, settings, speaker_count=args.speaker_count
        )
        summaries.append(summary)
        print(
            json.dumps({k: v for k, v in summary.items() if k != "windows"}),
            flush=True,
        )

    PRIVATE.mkdir(parents=True, exist_ok=True)
    payload: dict = {"summaries": summaries}
    if args.final_meeting_id:
        for summary in summaries:
            summary["final_comparison"] = final_comparison(args.final_meeting_id, summary)
        print(
            json.dumps(
                {s["config"]: s["final_comparison"] for s in summaries},
                ensure_ascii=False,
            ),
            flush=True,
        )
    (PRIVATE / "diarization-diagnostic.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"private details: {PRIVATE / 'diarization-diagnostic.json'}", flush=True)

    # Phase 5 rollup: how often a window even contains multiple speakers.
    for summary in summaries:
        print(
            f"phase5 {summary['config']}: "
            f"multi_provider_windows={summary['windows_with_multiple_provider_speakers']} "
            f"single_provider_windows={summary['windows_with_single_provider_speaker']} "
            f"empty_windows={summary['windows_without_words']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
