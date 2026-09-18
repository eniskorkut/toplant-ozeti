#!/usr/bin/env python3
"""Build the safe comparison aggregate for the ElevenLabs benchmark (host, stdlib).

Reads the raw responses and the pre-computed local spot-check, compares against the
reference RTTMs with the same best-permutation methodology as the local benchmarks, and
writes an aggregate that contains NO transcript text, no key and no headers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO_ROOT = HARNESS.parent.parent
RESULTS = HARNESS / "results"
RAW = RESULTS / "raw"

sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "overlap-merge"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "stt"))

from el_parse import parse_words, response_metrics  # noqa: E402
from evaluate import evaluate_assignment, parse_rttm  # noqa: E402
from normalize import score as wer_score  # noqa: E402

VOXCONVERSE_RTTM = REPO_ROOT / "benchmarks" / "diarization" / "datasets" / "voxconverse" / "voxconverse" / "dev"
LOCAL_STT_RAW = REPO_ROOT / "benchmarks" / "overlap-merge" / "results" / "raw"
LOCAL_SPOT_CHECK = RESULTS / "local-spot-check.json"

TURKISH_LOCAL = {"far": 0.1429, "near": 0.2024, "combined": 0.1726}
VOXCONVERSE_FILES = {"whmpa": "voxconverse-whmpa", "wjhgf": "voxconverse-wjhgf"}


def load_response(label: str) -> dict:
    return json.loads((RAW / f"{label}.json").read_text(encoding="utf-8"))


def elevenlabs_spot_check() -> dict:
    output: dict = {}
    reference_text = (REPO_ROOT / "benchmarks" / "stt" / "reference_tr.txt").read_text(encoding="utf-8")

    for file_id, label in VOXCONVERSE_FILES.items():
        payload = load_response(label)
        words = parse_words(payload)
        reference = parse_rttm(VOXCONVERSE_RTTM / f"{file_id}.rttm")
        evaluation = evaluate_assignment(
            words,
            [word.speaker for word in words],
            reference,
        )
        output[file_id] = {
            "detected_speakers": response_metrics(payload)["distinct_speakers"],
            "agreement": evaluation["agreement"],
            "wrong_attribution_rate": evaluation["wrong_attribution_rate"],
            "wrong_attribution_words": evaluation["wrong_attribution_words"],
            "untagged_words": response_metrics(payload)["untagged_words"],
            "unresolved_rate": response_metrics(payload)["untagged_words"] / max(len(words), 1),
            "turns": response_metrics(payload)["speaker_turns"],
            "rapid_flips": response_metrics(payload)["rapid_flips"],
            "compared_words": evaluation["compared_words"],
        }

    turkish: dict = {}
    totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    for name, label in (("far", "turkish-far"), ("near", "turkish-near")):
        payload = load_response(label)
        text = " ".join(word.text for word in parse_words(payload))
        result = wer_score(reference_text, text)
        turkish[name] = {
            "wer": round(result["wer"], 4),
            "substitutions": result["substitutions"],
            "deletions": result["deletions"],
            "insertions": result["insertions"],
            "hypothesis_words": result["hypothesis_words"],
        }
        for key in totals:
            totals[key] += result[key]
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    turkish["combined"] = {
        **totals,
        "total_errors": errors,
        "wer": round(errors / totals["reference_words"], 4),
    }

    local_turkish: dict = {}
    for name, recording in (("far", "sample_far"), ("near", "sample_near")):
        payload = json.loads((LOCAL_STT_RAW / f"{recording}--heuristic.json").read_text(encoding="utf-8"))
        result = wer_score(reference_text, payload["text"])
        local_turkish[name] = {"wer": round(result["wer"], 4), "words": payload["word_count"]}
    local_errors = sum(local_turkish[name]["wer"] for name in ("far", "near"))
    local_turkish["combined_wer"] = round(local_errors / 2, 4)

    return {"voxconverse": output, "turkish": {"elevenlabs": turkish, "local_heuristic": local_turkish}}


def real_meeting_metrics() -> dict:
    known4 = response_metrics(load_response("real-known4"))
    auto = response_metrics(load_response("real-auto"))
    guard = json.loads((RESULTS / "request-guard.json").read_text(encoding="utf-8"))
    latencies = {call["label"]: call["latency_seconds"] for call in guard["calls"]}

    with (LOCAL_STT_RAW / "b1095740120b4b1e96db337da961ea63--heuristic.json").open() as handle:
        local_stt = json.load(handle)

    return {
        "local_production": {"words": 106, "assigned": 85, "unresolved": 21, "speakers": 4, "turns": 13, "rapid_flips": 0},
        "local_tuned_0_0": {"words": 106, "assigned": 91, "unresolved": 15, "speakers": 4, "turns": 17, "rapid_flips": 3},
        "elevenlabs_known4": {
            "words": known4["words"],
            "assigned": known4["speaker_tagged_words"],
            "unresolved": known4["untagged_words"],
            "speakers": known4["distinct_speakers"],
            "turns": known4["speaker_turns"],
            "rapid_flips": known4["rapid_flips"],
            "latency_seconds": latencies.get("real-known4"),
        },
        "elevenlabs_auto": {
            "words": auto["words"],
            "assigned": auto["speaker_tagged_words"],
            "unresolved": auto["untagged_words"],
            "speakers": auto["distinct_speakers"],
            "turns": auto["speaker_turns"],
            "rapid_flips": auto["rapid_flips"],
            "latency_seconds": latencies.get("real-auto"),
        },
        "local_stt_decoding_seconds": local_stt["decoding_seconds"],
        "local_diarization_rtf_reference": 0.084,
    }


def markdown(aggregate: dict) -> str:
    lines = ["# ElevenLabs Scribe v2 vs local pipeline", ""]
    turkish = aggregate["turkish"]
    lines += [
        "## Turkish WER (controlled recordings)",
        "",
        "| Recording | Local whisper.cpp | ElevenLabs |",
        "|---|---:|---:|",
        f"| far | {turkish['local_heuristic']['far']['wer']:.4f} | {turkish['elevenlabs']['far']['wer']:.4f} |",
        f"| near | {turkish['local_heuristic']['near']['wer']:.4f} | {turkish['elevenlabs']['near']['wer']:.4f} |",
        f"| combined | {turkish['local_heuristic']['combined_wer']:.4f} | {turkish['elevenlabs']['combined']['wer']:.4f} |",
        "",
        "## Real four-speaker meeting (no temporal ground truth)",
        "",
        "| Metric | Local Prod 0.3/0.5 | Local Tuned 0/0 | ElevenLabs known4 | ElevenLabs auto |",
        "|---|---:|---:|---:|---:|",
    ]
    real = aggregate["real_meeting"]
    for field, label in (
        ("words", "words"),
        ("assigned", "speaker-assigned"),
        ("unresolved", "unassigned"),
        ("speakers", "speakers"),
        ("turns", "turns"),
        ("rapid_flips", "rapid flips"),
    ):
        lines.append(
            f"| {label} | {real['local_production'][field]} | {real['local_tuned_0_0'][field]} | "
            f"{real['elevenlabs_known4'][field]} | {real['elevenlabs_auto'][field]} |"
        )
    lines.append(
        f"| latency | STT {real['local_stt_decoding_seconds']} s + diarization | same | "
        f"{real['elevenlabs_known4']['latency_seconds']} s | {real['elevenlabs_auto']['latency_seconds']} s |"
    )
    lines += [
        "",
        "## VoxConverse limited two-file spot check",
        "",
        "| File | Metric | Local Prod | Local Tuned | ElevenLabs |",
        "|---|---|---:|---:|---:|",
    ]
    for file_id, data in aggregate["voxconverse"].items():
        local = aggregate["local_spot_check"][file_id]
        rows = (
            ("detected_speakers", "detected_speakers", "detected speakers"),
            ("agreement", "agreement", "speaker agreement"),
            ("wrong_attribution_rate", "wrong_attribution_rate", "wrong attribution"),
            ("untagged_words", "unresolved_words", "untagged/unresolved"),
            ("turns", "turns", "turns"),
            ("rapid_flips", "rapid_flips", "rapid flips"),
        )
        for eleven_field, local_field, label in rows:
            sixty = local["on0.30_off0.50"].get(local_field)
            tuned = local["on0.00_off0.00"].get(local_field)
            eleven = data.get(eleven_field)
            lines.append(f"| {file_id} | {label} | {sixty} | {tuned} | {eleven} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    aggregate = {
        "provider": "elevenlabs",
        "model": "scribe_v2",
        "requests": json.loads((RESULTS / "request-guard.json").read_text(encoding="utf-8")),
        "retention_note": "normal ElevenLabs retention may apply (free/dev account; no zero-retention mode requested)",
        **elevenlabs_spot_check(),
        "real_meeting": real_meeting_metrics(),
        "local_spot_check": json.loads(LOCAL_SPOT_CHECK.read_text(encoding="utf-8")),
        "local_turkish_reference": TURKISH_LOCAL,
    }
    (HARNESS / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS / "report.md").write_text(markdown(aggregate), encoding="utf-8")
    print(markdown(aggregate))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
