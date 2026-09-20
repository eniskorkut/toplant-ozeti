#!/usr/bin/env python3
"""Conservative-confirmation policy study for live speaker attribution.

Runs the accepted 12 s / 4 s snapshot schedule on existing recordings (no new
recordings, no threshold tuning), records every stable matched span with its
confidence inputs (score ratio, best-vs-second margin, snapshot coexistence,
window end) and evaluates confirmation policies against each recording's own
authoritative full-file diarization.

Policies (evaluated offline from the same recorded evidence):

  P1  current: any stable matched span confirms
  P2  >=2 agreeing snapshots AND margin >= M
  P3  >=3 agreeing snapshots OR (>=2 agreeing AND provider speakers coexisted)
  P4  strong majority (>= 0.75) across all snapshots covering the instant

Structural data only: no transcript text, no raw provider payloads.
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
GUARD_FILE = SCRIPT_DIR / "results" / "conservative-guard.json"
MAX_REQUESTS = 40

LOOKBACK = 12.0
STEP = 4.0
FIRST_ATTEMPT = 8.0
RESOLUTION = 0.25
PROVIDER_LATENCY_SECONDS = 1.0  # measured median; only used for delay estimates

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


def collect_history(meeting_id: str, pcm: bytes, duration: float, settings: Settings) -> dict:
    rb.GUARD_FILE = GUARD_FILE
    rb.MAX_REQUESTS = MAX_REQUESTS
    store = LiveSessionStore(ttl_seconds=3600)
    session = store.create()
    latencies: list[float] = []
    uploaded = 0.0
    for sequence, (start, end) in enumerate(snapshot_schedule(duration), start=1):
        window_pcm = pcm[int(start * 32_000): int(end * 32_000)]
        guard = rb.load_guard()
        guard.check()
        started = time.perf_counter()
        words = transcribe_pcm_window(window_pcm, settings=settings, requested_speaker_count=None)
        latency = time.perf_counter() - started
        guard.record(label=f"{meeting_id}-w{sequence}", audio_seconds=end - start,
                     latency_seconds=latency, status=200)
        rb.save_guard(guard)
        latencies.append(latency)
        uploaded += end - start
        store.apply_window(
            session,
            provider_intervals=provider_intervals(words, start, (start, end)),
            window=(start, end),
            sequence=sequence,
            stable_until=max(start, end - STEP),
        )
    with httpx.Client(base_url=API, timeout=30.0) as client:
        turns = client.get(f"/api/v1/meetings/{meeting_id}/transcript").json()["turns"]
    return {
        "duration": duration,
        "history": session.assignment_history,
        "turns": turns,
        "requests": len(snapshot_schedule(duration)),
        "uploaded_audio_seconds": round(uploaded, 1),
        "median_latency": round(statistics.median(latencies), 3) if latencies else None,
        "confirmed_speakers": session.confirmed_labels(),
    }


def votes_at(history: list[dict], instant: float) -> list[dict]:
    """One vote per snapshot sequence covering the instant (newest wins per seq)."""
    by_sequence: dict[int, dict] = {}
    for entry in history:
        if entry["start"] <= instant <= entry["end"]:
            by_sequence[entry["sequence"]] = entry  # later spans override same seq
    return list(by_sequence.values())


def policy_label(votes: list[dict], policy: str, margin_threshold: float) -> tuple[str | None, int]:
    if not votes:
        return None, 0
    counts: dict[str, list[dict]] = {}
    for vote in votes:
        counts.setdefault(vote["label"], []).append(vote)
    best_label = max(counts, key=lambda label: (len(counts[label]), label))
    supporting = counts[best_label]
    if policy == "P1":
        return best_label, len(supporting)
    if policy == "P2":
        margin = max(entry["margin"] for entry in supporting)
        if len(supporting) >= 2 and margin >= margin_threshold:
            return best_label, len(supporting)
        return None, 0
    if policy == "P3":
        coexist = any(entry["coexist"] for entry in supporting)
        if len(supporting) >= 3 or (len(supporting) >= 2 and coexist):
            return best_label, len(supporting)
        return None, 0
    if policy == "P4":
        agreement = len(supporting) / len(votes)
        if len(votes) >= 2 and agreement >= 0.75:
            return best_label, len(supporting)
        return None, 0
    if policy == "P5":
        # Provider itself distinguished multiple speakers in a supporting snapshot.
        coexist_count = sum(1 for entry in supporting if entry["coexist"])
        if len(supporting) >= 2 and coexist_count >= 1:
            return best_label, len(supporting)
        return None, 0
    if policy == "P6":
        coexist_count = sum(1 for entry in supporting if entry["coexist"])
        if len(supporting) >= 2 and coexist_count >= 2:
            return best_label, len(supporting)
        return None, 0
    if policy == "P8":
        # Only unambiguous evidence may confirm: a cluster that overlapped the
        # previous snapshot, or a continuation in a snapshot where the provider
        # saw a single speaker. Merged/split clusters never confirm.
        strong = [
            entry
            for entry in supporting
            if entry["evidence"] == "overlap"
            or (entry["evidence"] == "gap" and not entry["coexist"])
        ]
        if len(supporting) >= 2 and len(strong) >= 2:
            return best_label, len(supporting)
        return None, 0
    if policy == "P9":
        strong = [
            entry
            for entry in supporting
            if entry["evidence"] == "overlap"
            or (entry["evidence"] == "gap" and not entry["coexist"])
        ]
        margin = max(entry["margin"] for entry in supporting)
        if len(supporting) >= 2 and len(strong) >= 2 and margin >= margin_threshold:
            return best_label, len(supporting)
        return None, 0
    if policy == "P7":
        coexist_count = sum(1 for entry in supporting if entry["coexist"])
        margin = max(entry["margin"] for entry in supporting)
        if len(supporting) >= 2 and coexist_count >= 1 and margin >= margin_threshold:
            return best_label, len(supporting)
        return None, 0
    if policy == "P10":
        # P7 with promoted speakers exempt from the overlap-margin requirement
        # (they were promoted by multi-snapshot evidence, not overlap scores).
        coexist_count = sum(1 for entry in supporting if entry["coexist"])
        margins = [
            entry["margin"] for entry in supporting if entry["evidence"] != "promoted"
        ]
        margin = max(margins) if margins else float("inf")
        if len(supporting) >= 2 and coexist_count >= 1 and margin >= margin_threshold:
            return best_label, len(supporting)
        return None, 0
    raise SystemExit(f"unknown policy {policy}")


def optimal_mapping(turn_seconds_by_final: dict[str, dict[str, float]]) -> dict[str, str]:
    """One-to-one final speaker -> canonical by matched seconds (greedy optimal)."""
    pairs = [
        (seconds, final_speaker, label)
        for final_speaker, labels in turn_seconds_by_final.items()
        for label, seconds in labels.items()
    ]
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    mapping: dict[str, str] = {}
    used_labels: set[str] = set()
    for _seconds, final_speaker, label in pairs:
        if final_speaker in mapping or label in used_labels:
            continue
        mapping[final_speaker] = label
        used_labels.add(label)
    return mapping


def evaluate(data: dict, policy: str, margin_threshold: float) -> dict:
    history = data["history"]
    turns = data["turns"]
    duration = data["duration"]

    timeline: dict[float, tuple[str | None, int]] = {}
    for index in range(int(duration / RESOLUTION) + 1):
        instant = index * RESOLUTION
        timeline[instant] = policy_label(votes_at(history, instant), policy, margin_threshold)

    def grid(instant: float) -> float:
        return round(round(instant / RESOLUTION) * RESOLUTION, 2)

    per_final: dict[str, dict[str, float]] = {}
    correct = wrong = provisional = 0.0
    for turn in turns:
        if turn["speaker"] == "Bilinmeyen":
            continue
        speaker = turn["speaker"]
        label_seconds: dict[str | None, float] = {}
        instant = turn["start_seconds"]
        while instant < turn["end_seconds"]:
            label, _support = timeline.get(grid(instant), (None, 0))
            label_seconds[label] = label_seconds.get(label, 0.0) + RESOLUTION
            instant += RESOLUTION
        per_final.setdefault(speaker, {})
        for label, seconds in label_seconds.items():
            if label is not None:
                per_final[speaker][label] = per_final[speaker].get(label, 0.0) + seconds

    mapping = optimal_mapping(per_final)
    # seconds per final speaker mapped correctly / wrongly / provisional
    correct_seconds = wrong_seconds = provisional_seconds = 0.0
    for turn in turns:
        if turn["speaker"] == "Bilinmeyen":
            continue
        speaker = turn["speaker"]
        expected_label = mapping.get(speaker)
        instant = turn["start_seconds"]
        while instant < turn["end_seconds"]:
            label, _support = timeline.get(grid(instant), (None, 0))
            if label is None:
                provisional_seconds += RESOLUTION
            elif label == expected_label:
                correct_seconds += RESOLUTION
            else:
                wrong_seconds += RESOLUTION
            instant += RESOLUTION

    confirmed_seconds = correct_seconds + wrong_seconds
    precision = correct_seconds / confirmed_seconds if confirmed_seconds else 0.0

    # confirmation delay: first supporting snapshot end + provider latency - turn end
    delays: list[float] = []
    for turn in turns:
        if turn["speaker"] == "Bilinmeyen":
            continue
        midpoint = (turn["start_seconds"] + turn["end_seconds"]) / 2
        votes = votes_at(history, midpoint)
        label, _support = policy_label(votes, policy, margin_threshold)
        if label is None:
            continue
        supporting = [vote for vote in votes if vote["label"] == label]
        if not supporting:
            continue
        first_window_end = min(entry["window_end"] for entry in supporting)
        delays.append(max(0.0, first_window_end + PROVIDER_LATENCY_SECONDS - turn["end_seconds"]))

    # label switches on the consensus timeline (ignoring provisional gaps)
    switches = 0
    previous: str | None = None
    for index in range(len(timeline)):
        label, _ = timeline[index * RESOLUTION]
        if label is None:
            continue
        if previous is not None and label != previous:
            switches += 1
        previous = label

    return {
        "policy": policy,
        "margin_threshold": margin_threshold,
        "precision": round(precision, 3),
        "wrong_confirmed_seconds": round(wrong_seconds, 1),
        "correct_confirmed_seconds": round(correct_seconds, 1),
        "confirmed_coverage": round(confirmed_seconds / max(correct_seconds + wrong_seconds + provisional_seconds, 1e-6), 3),
        "provisional_rate": round(provisional_seconds / max(correct_seconds + wrong_seconds + provisional_seconds, 1e-6), 3),
        "canonical_speakers": sorted(per_final and set().union(*[set(v) for v in per_final.values()]) or set()),
        "switches": switches,
        "median_confirmation_delay_seconds": round(statistics.median(delays), 2) if delays else None,
        "p95_confirmation_delay_seconds": round(sorted(delays)[int(len(delays) * 0.95)], 2)
        if delays
        else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-thresholds", default="0.5,1.0,2.0")
    parser.add_argument("--reuse", action="store_true", help="evaluate stored history only")
    args = parser.parse_args()

    settings = Settings()
    report: dict = {"recordings": {}, "policies": {}}
    datasets: dict[str, dict] = {}
    stored = {}
    stored_path = PRIVATE / "conservative-calibration.json"
    if args.reuse:
        if not stored_path.is_file():
            raise SystemExit("no stored calibration data to reuse")
        stored = json.loads(stored_path.read_text(encoding="utf-8")).get("recordings", {})
    for name, spec in RECORDINGS.items():
        if args.reuse:
            data = {
                "history": stored[name]["history"],
                "turns": stored[name]["turns"],
                "duration": stored[name]["duration"],
                "requests": stored[name]["requests"],
                "uploaded_audio_seconds": stored[name]["uploaded_audio_seconds"],
                "median_latency": stored[name]["median_latency"],
            }
        else:
            audio = rb.REPO_ROOT / "data" / "meetings" / spec["meeting_id"] / "processing.wav"
            if not audio.is_file():
                raise SystemExit(f"audio missing: {audio}")
            pcm, duration = read_pcm(audio)
            data = collect_history(spec["meeting_id"], pcm, duration, settings)
        datasets[name] = data
        margins = sorted(entry["margin"] for entry in data["history"])
        quantiles = {
            "p25": round(margins[int(len(margins) * 0.25)], 2) if margins else None,
            "p50": round(margins[int(len(margins) * 0.5)], 2) if margins else None,
            "p75": round(margins[int(len(margins) * 0.75)], 2) if margins else None,
        }
        report["recordings"][name] = {
            "history": data["history"],
            "turns": data["turns"],
            "duration": data["duration"],
            "requests": data["requests"],
            "uploaded_audio_seconds": data["uploaded_audio_seconds"],
            "median_latency": data["median_latency"],
            "stable_spans": len(data["history"]),
            "margin_quantiles": quantiles,
            "final_speakers": sorted({turn["speaker"] for turn in data["turns"]}),
        }
        print(f"== {name}: requests={data['requests']} spans={len(data['history'])} margins={quantiles}",
              flush=True)

    thresholds = [float(value) for value in args.margin_thresholds.split(",")]
    policies = ["P1", "P3", "P5", "P6", "P8"] + [
        f"P2@{threshold}" for threshold in thresholds
    ] + [f"P7@{threshold}" for threshold in thresholds] + [
        f"P9@{threshold}" for threshold in thresholds
    ]
    for policy in policies:
        base, threshold = (policy.split("@") + ["0"])[:2]
        per_recording = {}
        for name, data in datasets.items():
            per_recording[name] = evaluate(data, base, float(threshold))
        combined = {
            "precision": round(
                sum(r["precision"] * (r["correct_confirmed_seconds"] + r["wrong_confirmed_seconds"])
                    for r in per_recording.values())
                / max(sum(r["correct_confirmed_seconds"] + r["wrong_confirmed_seconds"]
                          for r in per_recording.values()), 1e-6), 3),
            "wrong_confirmed_seconds": round(sum(r["wrong_confirmed_seconds"] for r in per_recording.values()), 1),
            "confirmed_coverage": round(statistics.mean(r["confirmed_coverage"] for r in per_recording.values()), 3),
            "provisional_rate": round(statistics.mean(r["provisional_rate"] for r in per_recording.values()), 3),
            "median_confirmation_delay_seconds": statistics.median(
                r["median_confirmation_delay_seconds"] for r in per_recording.values()
                if r["median_confirmation_delay_seconds"] is not None
            ),
        }
        report["policies"][policy] = {"combined": combined, "per_recording": per_recording}
        print(json.dumps({policy: {**combined, "per_recording": per_recording}}, ensure_ascii=False),
              flush=True)

    if not args.reuse:
        PRIVATE.mkdir(parents=True, exist_ok=True)
        (PRIVATE / "conservative-calibration.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(f"private details: {PRIVATE / 'conservative-calibration.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
