#!/usr/bin/env python3
"""Diarization post-processing sweep: runs TitaNet for a min_duration_on/off grid.

Runs inside the backend image. For every configuration and file it stores the
diarization segments, writes a system RTTM for the isolated dscore scoring step, and
computes coverage/overlap/bucket metrics plus production-merge diagnostics.

The runtime database, the real meeting row/transcript and production settings are never
modified.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, "/bench")
sys.path.insert(0, "/overlap")

from dp_metrics import (  # noqa: E402
    Segment,
    coverage_metrics,
    overlap_metrics,
    segment_duration_buckets,
    unresolved_causes,
)
from evaluate import evaluate_assignment, parse_rttm, speaker_metrics  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.diarization import diarize  # noqa: E402
from app.services.merge import DiarizationSegment, Word, assign_speaker  # noqa: E402

HARNESS = Path("/bench")
RESULTS = HARNESS / "results"
STT_RAW = Path("/stt-raw")
VOXCONVERSE = Path("/voxconverse")

CALIBRATION = ["qpylu", "fxgvy", "szsyz", "rtvuw", "gwtwd", "bwzyf"]
VALIDATION = ["whmpa", "bkwns", "syiwe", "jiqvr", "jyirt", "wjhgf"]
REAL_MEETING = "b1095740120b4b1e96db337da961ea63"

GRID_ON = [0.0, 0.1, 0.2, 0.3]
GRID_OFF = [0.0, 0.10, 0.25, 0.50]

THRESHOLD = 0.80
THREADS = 8
MERGE_TOLERANCE = 0.25


def log(message: str) -> None:
    print(message, flush=True)


def config_key(on: float, off: float) -> str:
    return f"on{on:.2f}_off{off:.2f}"


def audio_path(file_id: str) -> Path:
    if file_id == REAL_MEETING:
        return Path(f"/data/meetings/{file_id}/processing.wav")
    return VOXCONVERSE / "audio" / f"{file_id}.wav"


def reference_path(file_id: str) -> Path | None:
    if file_id == REAL_MEETING:
        return None
    return VOXCONVERSE / "voxconverse" / "dev" / f"{file_id}.rttm"


def stt_words(file_id: str) -> tuple[list[Word], float]:
    cache = STT_RAW / f"{file_id}--heuristic.json"
    payload = json.loads(cache.read_text(encoding="utf-8"))
    words = [Word(word["start"], word["end"], word["text"]) for word in payload["words"]]
    return words, payload["audio_seconds"]


def write_system_rttm(path: Path, file_id: str, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"SPEAKER {file_id} 1 {segment.start:.3f} {segment.end - segment.start:.3f} "
        f"<NA> <NA> speaker_{segment.speaker} <NA> <NA>"
        for segment in segments
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibration", "validation", "real"], required=True)
    parser.add_argument(
        "--configs",
        default="",
        help="comma separated on/off pairs, e.g. '0.3/0.5,0.1/0.25'; default = full grid for calibration",
    )
    args = parser.parse_args()

    if args.configs:
        configs = [
            (float(pair.split("/")[0]), float(pair.split("/")[1]))
            for pair in args.configs.split(",")
        ]
    else:
        configs = [(on, off) for on in GRID_ON for off in GRID_OFF]

    files = {
        "calibration": CALIBRATION,
        "validation": VALIDATION,
        "real": [REAL_MEETING],
    }[args.stage]

    results: dict = {
        "stage": args.stage,
        "files": files,
        "configs": {},
        "grid": {"min_duration_on": GRID_ON, "min_duration_off": GRID_OFF},
        "fixed": {"threshold": THRESHOLD, "threads": THREADS, "merge_tolerance": MERGE_TOLERANCE},
    }

    for on, off in configs:
        key = config_key(on, off)
        log(f"config {key}")
        settings = Settings(
            diarization_threshold=THRESHOLD,
            diarization_threads=THREADS,
            diarization_min_duration_on=on,
            diarization_min_duration_off=off,
        )
        aggregate = {
            "min_duration_on": on,
            "min_duration_off": off,
            "config": key,
            "segments": 0,
            "coverage_seconds": 0.0,
            "audio_seconds": 0.0,
            "coverage_ratio": 0.0,
            "uncovered_gap_seconds": 0.0,
            "longest_uncovered_gap_seconds": 0.0,
            "overlap_seconds": 0.0,
            "overlap_ratio": 0.0,
            "buckets": {"lt_0.3s": 0, "0.3_to_0.5s": 0, "0.5_to_1.0s": 0, "gt_1.0s": 0},
            "total_words": 0,
            "assigned_words": 0,
            "unresolved_words": 0,
            "wrong_attribution_words": 0,
            "compared_words": 0,
            "speaker_turns": 0,
            "rapid_flips": 0,
            "detected_speakers": [],
            "reference_speakers": [],
            "unresolved_causes": {"overlap_ambiguity": 0, "coverage_gap": 0, "other": 0},
        }

        for file_id in files:
            result = diarize(
                audio_path(file_id), settings, requested_speaker_count=None
            )
            segments = [
                Segment(segment.start, segment.end, str(segment.speaker))
                for segment in result.segments
            ]
            payload = {
                "file_id": file_id,
                "num_speakers": result.num_speakers,
                "segments": [
                    {"start": segment.start, "end": segment.end, "speaker": segment.speaker}
                    for segment in segments
                ],
            }
            (RESULTS / "segments" / args.stage / key).mkdir(parents=True, exist_ok=True)
            (RESULTS / "segments" / args.stage / key / f"{file_id}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            write_system_rttm(RESULTS / "rttm" / args.stage / key / f"{file_id}.rttm", file_id, segments)

            with wave.open(str(audio_path(file_id)), "rb") as wav_file:
                audio_seconds = wav_file.getnframes() / wav_file.getframerate()

            coverage = coverage_metrics(segments, audio_seconds)
            overlap = overlap_metrics(segments, audio_seconds)
            buckets = segment_duration_buckets(segments)
            words, _ = stt_words(file_id)
            production = [
                DiarizationSegment(segment.start, segment.end, segment.speaker)
                for segment in segments
            ]
            speakers = [assign_speaker(word, production, MERGE_TOLERANCE) for word in words]
            causes = unresolved_causes(words, speakers, segments)
            turn_metrics = speaker_metrics(words, speakers)

            aggregate["segments"] += len(segments)
            aggregate["coverage_seconds"] += coverage["coverage_seconds"]
            aggregate["audio_seconds"] += audio_seconds
            aggregate["uncovered_gap_seconds"] += coverage["uncovered_gap_seconds"]
            aggregate["longest_uncovered_gap_seconds"] = max(
                aggregate["longest_uncovered_gap_seconds"], coverage["longest_uncovered_gap_seconds"]
            )
            aggregate["overlap_seconds"] += overlap["overlap_seconds"]
            for bucket, value in buckets.items():
                aggregate["buckets"][bucket] += value
            aggregate["total_words"] += len(words)
            aggregate["assigned_words"] += sum(1 for speaker in speakers if speaker is not None)
            aggregate["unresolved_words"] += sum(1 for speaker in speakers if speaker is None)
            aggregate["speaker_turns"] += turn_metrics["turns"]
            aggregate["rapid_flips"] += turn_metrics["rapid_flips"]
            aggregate["detected_speakers"].append(result.num_speakers)
            for cause, value in causes.items():
                aggregate["unresolved_causes"][cause] += value

            reference = reference_path(file_id)
            if reference is not None:
                evaluation = evaluate_assignment(
                    words,
                    [f"speaker_{speaker}" if speaker is not None else None for speaker in speakers],
                    parse_rttm(reference),
                )
                aggregate["wrong_attribution_words"] += evaluation["wrong_attribution_words"]
                aggregate["compared_words"] += evaluation["compared_words"]
                aggregate["reference_speakers"].append(
                    len({turn.speaker for turn in parse_rttm(reference)})
                )

        total_words = aggregate["total_words"] or 1
        aggregate["coverage_ratio"] = round(
            aggregate["coverage_seconds"] / aggregate["audio_seconds"], 4
        )
        aggregate["overlap_ratio"] = round(aggregate["overlap_seconds"] / aggregate["audio_seconds"], 4)
        aggregate["unresolved_rate"] = round(aggregate["unresolved_words"] / total_words, 4)
        aggregate["wrong_attribution_rate"] = (
            round(aggregate["wrong_attribution_words"] / aggregate["compared_words"], 4)
            if aggregate["compared_words"]
            else 0.0
        )
        if aggregate["reference_speakers"]:
            errors = [
                abs(detected - reference)
                for detected, reference in zip(
                    aggregate["detected_speakers"], aggregate["reference_speakers"], strict=True
                )
            ]
            aggregate["speaker_count_mae"] = round(sum(errors) / len(errors), 4)
            aggregate["speaker_count_accuracy"] = round(
                sum(1 for error in errors if error == 0) / len(errors), 4
            )
        else:
            aggregate["speaker_count_mae"] = None
            aggregate["speaker_count_accuracy"] = None

        results["configs"][key] = aggregate
        log(
            f"  segments {aggregate['segments']} | coverage {aggregate['coverage_ratio']:.1%} | "
            f"overlap {aggregate['overlap_ratio']:.1%} | unresolved {aggregate['unresolved_words']} | "
            f"wrong {aggregate['wrong_attribution_words']}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"metrics-{args.stage}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote {RESULTS / f'metrics-{args.stage}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
