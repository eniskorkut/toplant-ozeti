#!/usr/bin/env python3
"""Controlled A/B/C/D diagnostic on ONE existing real four-speaker meeting.

Runs INSIDE the backend image (whisper-cli + sherpa-onnx + mounted models) and calls
the production services directly, overriding only language and speaker count:

    A = language auto + automatic speakers (threshold 0.80)
    B = language tr   + automatic speakers (threshold 0.80)
    C = language auto + known speakers = N
    D = language tr   + known speakers = N

The recording is opened read-only; nothing in the runtime database, transcript,
analysis or media files is touched. Transcript-bearing outputs go to
`output/` (git-ignored); only aggregate metrics without transcript text are written
next to this script.

Usage (from the repository root):

    docker compose run --rm \
        -v "$PWD/benchmarks/real-meeting-diagnostic:/bench" \
        backend uv run --locked python /bench/run_diagnostic.py \
            --meeting-id <id> --audio /data/meetings/<id>/processing.wav --real-speakers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/bench")

from diagnostic_metrics import (  # noqa: E402
    diarization_metrics,
    merge_metrics,
    normalize_words,
    segment_coverage,
    short_examples,
    text_delta,
    timestamp_health,
    unresolved_causes,
    unresolved_word_gaps,
)

from app.config import Settings  # noqa: E402
from app.services.diarization import diarize  # noqa: E402
from app.services.merge import DiarizationSegment, Word, merge_words  # noqa: E402
from app.services.stt import transcribe  # noqa: E402

CONFIGS = [
    {"label": "A", "language": "auto", "speaker_count": None, "name": "A-auto-auto"},
    {"label": "B", "language": "tr", "speaker_count": None, "name": "B-tr-auto"},
    {"label": "C", "language": "auto", "speaker_count": 4, "name": "C-auto-known4"},
    {"label": "D", "language": "tr", "speaker_count": 4, "name": "D-tr-known4"},
]


def log(message: str) -> None:
    print(message, flush=True)


def write_transcript(path: Path, turns: list) -> None:
    lines = [
        f"[{turn.start:8.3f}] {turn.speaker}: {turn.text}" for turn in turns
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_diff(path: Path, auto_text: str, tr_text: str) -> None:
    auto_words = normalize_words(auto_text)
    tr_words = normalize_words(tr_text)
    diff = [
        f"auto words: {len(auto_words)} | tr words: {len(tr_words)}",
        "",
        "word-level differences (auto -> tr):",
    ]
    import difflib

    matcher = difflib.SequenceMatcher(a=auto_words, b=tr_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        auto_span = " ".join(auto_words[i1:i2]) or "∅"
        tr_span = " ".join(tr_words[j1:j2]) or "∅"
        diff.append(f"  {tag:8} auto[{i1}:{i2}] {auto_span!r} -> tr[{j1}:{j2}] {tr_span!r}")
    path.write_text("\n".join(diff) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meeting-id", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--real-speakers", type=int, default=None)
    parser.add_argument("--output-dir", default="/bench/output")
    parser.add_argument("--repo-dir", default="/bench")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"audio input missing: {audio_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        stt_threads=8,
        stt_beam_size=5,
        stt_language="auto",
        diarization_threads=8,
        diarization_threshold=0.80,
        diarization_min_duration_on=0.3,
        diarization_min_duration_off=0.5,
        merge_boundary_tolerance_seconds=0.25,
    )

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meeting_id": args.meeting_id,
        "audio": {
            "path": str(audio_path),
            "real_speakers": args.real_speakers,
        },
        "production_overrides": "language and speaker count only; all other settings at production defaults",
        "configs": {},
    }
    texts: dict[str, str] = {}

    for config in CONFIGS:
        log(f"config {config['label']}: language={config['language']} speaker_count={config['speaker_count']}")
        run_settings = settings.model_copy(update={"stt_language": config["language"]})

        stt_result = transcribe(audio_path, run_settings)
        diarization_result = diarize(
            audio_path,
            run_settings,
            requested_speaker_count=config["speaker_count"],
        )
        merge_result = merge_words(
            [Word(word.start, word.end, word.text) for word in stt_result.words],
            [
                DiarizationSegment(segment.start, segment.end, segment.speaker)
                for segment in diarization_result.segments
            ],
            tolerance=run_settings.merge_boundary_tolerance_seconds,
        )

        audio_seconds = max(
            (stt_result.words[-1].end if stt_result.words else 0.0),
            (diarization_result.segments[-1].end if diarization_result.segments else 0.0),
        )
        texts[config["label"]] = stt_result.text

        write_transcript(output_dir / f"{config['name']}.txt", merge_result.turns)

        results["configs"][config["label"]] = {
            "label": config["label"],
            "language_requested": config["language"],
            "language_detected": stt_result.language,
            "speaker_count_requested": config["speaker_count"],
            "stt": {
                "transcript_chars": len(stt_result.text),
                "normalized_word_count": len(normalize_words(stt_result.text)),
                "inference_seconds": stt_result.inference_seconds,
                "rtf": round(stt_result.inference_seconds / audio_seconds, 4),
                **timestamp_health(stt_result.words),
            },
            "diarization": {
                **diarization_metrics(diarization_result.segments, audio_seconds),
                "reported_num_speakers": diarization_result.num_speakers,
                "inference_seconds": diarization_result.inference_seconds,
                "model_load_seconds": diarization_result.model_load_seconds,
                "rtf": round(diarization_result.inference_seconds / audio_seconds, 4),
            },
            "coverage": segment_coverage(diarization_result.segments, audio_seconds),
            "unresolved_word_gaps": unresolved_word_gaps(
                [
                    word
                    for turn in merge_result.turns
                    if turn.speaker == "Bilinmeyen"
                    for word in turn.word_list
                ],
                diarization_result.segments,
            ),
            "unresolved_causes": unresolved_causes(
                [
                    word
                    for turn in merge_result.turns
                    if turn.speaker == "Bilinmeyen"
                    for word in turn.word_list
                ],
                diarization_result.segments,
                settings.merge_boundary_tolerance_seconds,
            ),
            "merge": {
                "assigned_words": merge_result.assigned_words,
                "unresolved_words": merge_result.unresolved_words,
                **merge_metrics(
                    merge_result.turns, audio_seconds, real_speakers=args.real_speakers
                ),
            },
        }

    # Text comparison (auto vs tr) with a private, git-ignored diff for local review.
    write_diff(output_dir / "diff_A_vs_B.txt", texts["A"], texts["B"])
    write_diff(output_dir / "diff_C_vs_D.txt", texts["C"], texts["D"])

    results["text_comparison"] = {
        "auto_vs_tr_automatic": {
            **text_delta(normalize_words(texts["A"]), normalize_words(texts["B"])),
            "examples": short_examples(normalize_words(texts["A"]), normalize_words(texts["B"])),
        },
        "auto_vs_tr_known4": {
            **text_delta(normalize_words(texts["C"]), normalize_words(texts["D"])),
            "examples": short_examples(normalize_words(texts["C"]), normalize_words(texts["D"])),
        },
    }

    # The committed aggregate must contain NO transcript text: drop the short
    # observational snippets, they stay only in the private output directory.
    safe_results = json.loads(json.dumps(results))
    for comparison in safe_results["text_comparison"].values():
        comparison.pop("examples", None)

    aggregate_path = Path(args.repo_dir) / "aggregate.json"
    aggregate_path.write_text(
        json.dumps(safe_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "aggregate.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "aggregate.md").write_text(markdown_report(results), encoding="utf-8")
    log(f"wrote {aggregate_path} (metrics only, no transcript text)")
    log(f"wrote private transcript outputs to {output_dir}")
    return 0


def markdown_report(results: dict) -> str:
    lines = [
        "# Real four-speaker meeting diagnostic (A/B/C/D)",
        "",
        f"Meeting: `{results['meeting_id']}` · real speakers: {results['audio']['real_speakers']}",
        f"Generated: {results['generated_at']}",
        "",
        "| Config | Language | Spk mode | STT chars | Words | STT RTF | Diar clusters | Diar RTF | Kişi | Turns | Unresolved words | Bilinmeyen turns | Speaker changes | Rapid flips | Speaker count error |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in results["configs"].items():
        stt = data["stt"]
        diar = data["diarization"]
        merge = data["merge"]
        lines.append(
            f"| {label} | {data['language_requested']} (detected: {data['language_detected']}) | "
            f"{'known=4' if data['speaker_count_requested'] else 'auto'} | {stt['transcript_chars']} | "
            f"{stt['normalized_word_count']} | {stt['rtf']:.4f} | {diar['non_empty_clusters']} | "
            f"{diar['rtf']:.4f} | {merge['final_speaker_count']} | {merge['turn_count']} | "
            f"{merge['unresolved_words']} | {merge['unknown_turn_count']} | {merge['speaker_changes']} | "
            f"{merge['rapid_flips']} | {merge.get('speaker_count_error', 'n/a')} |"
        )
    comparison = results["text_comparison"]
    lines += [
        "",
        "## Text comparison (factual, no WER)",
        "",
        f"- automatic: {comparison['auto_vs_tr_automatic']['differing_word_positions']} differing word positions "
        f"of ~{comparison['auto_vs_tr_automatic']['auto_words']} words "
        f"(similarity {comparison['auto_vs_tr_automatic']['similarity_ratio']})",
        f"- known=4: {comparison['auto_vs_tr_known4']['differing_word_positions']} differing word positions "
        f"of ~{comparison['auto_vs_tr_known4']['auto_words']} words "
        f"(similarity {comparison['auto_vs_tr_known4']['similarity_ratio']})",
        "",
        "## Coverage and unresolved-word gaps",
        "",
        "| Config | Diarization coverage | Longest coverage gap | Unresolved words | Median gap | Max gap | <=250ms | 250ms-1s | 1-3s | >3s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, data in results["configs"].items():
        coverage = data["coverage"]
        gaps = data["unresolved_word_gaps"]
        buckets = gaps["buckets"]
        lines.append(
            f"| {label} | {coverage['coverage_ratio']:.1%} ({coverage['coverage_seconds']} s) | "
            f"{coverage['longest_gap_seconds']} s | {gaps['count']} | {gaps['median_gap_seconds']} s | "
            f"{gaps['max_gap_seconds']} s | {buckets.get('overlapping_or_within_tolerance', 0)} | "
            f"{buckets.get('tolerance_to_1s', 0)} | {buckets.get('one_to_3s', 0)} | "
            f"{buckets.get('over_3s', 0)} |"
        )
    lines += [
        "",
        "## Why words stayed unresolved",
        "",
        "| Config | Overlap region (2+ segments own the midpoint) | Coverage gap (> tolerance from any segment) | Boundary/tie |",
        "|---|---:|---:|---:|",
    ]
    for label, data in results["configs"].items():
        causes = data["unresolved_causes"]
        lines.append(
            f"| {label} | {causes['overlap_region']} | {causes['coverage_gap']} | "
            f"{causes['boundary_or_tie']} |"
        )
    lines += [
        "",
        "## Per-cluster diagnostics",
        "",
        "| Config | Per-cluster seconds | Per-cluster segments | Longest turn | Shortest turn | Long merged-exchange turns |",
        "|---|---|---|---:|---:|---|",
    ]
    for label, data in results["configs"].items():
        merge = data["merge"]
        lines.append(
            f"| {label} | {data['diarization']['per_cluster_seconds']} | "
            f"{data['diarization']['per_cluster_segments']} | {merge['longest_turn_seconds']} s | "
            f"{merge['shortest_non_empty_turn_seconds']} s | "
            f"{merge['long_merged_exchange_turns']['count']} "
            f"({merge['long_merged_exchange_turns']['durations']}) |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
