#!/usr/bin/env python3
"""Container-side analysis for the overlap-merge benchmark.

Runs inside the backend image: uses the PRODUCTION merge rules
(`app.services.merge.assign_speaker`) as the authoritative baseline, the benchmark-only
conservative resolver, and the VoxConverse reference RTTMs for calibration/validation.

Nothing here modifies production code, settings, the meeting row or the transcript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/bench")
sys.path.insert(0, "/stt-bench")

from context_resolver import Segment, apply_resolver  # noqa: E402
from evaluate import evaluate_assignment, parse_rttm, speaker_metrics  # noqa: E402

from app.config import Settings  # noqa: E402
from app.services.diarization import diarize  # noqa: E402
from app.services.merge import DiarizationSegment, Word, assign_speaker, label_speakers  # noqa: E402

HARNESS = Path("/bench")
RAW = HARNESS / "results" / "raw"
PRIVATE = HARNESS / "results" / "private"
VOXCONVERSE = Path("/voxconverse")
DIAR_RESULTS = Path("/diar-results")
MEETINGS = Path("/data/meetings")

CALIBRATION = ["qpylu", "fxgvy", "szsyz", "rtvuw", "gwtwd", "bwzyf"]
VALIDATION = ["whmpa", "bkwns", "syiwe", "jiqvr", "jyirt", "wjhgf"]
REAL_MEETING = "b1095740120b4b1e96db337da961ea63"
TURKISH_CONTROLLED = ["sample_far", "sample_near"]
WER_MODES = {"T0": "heuristic", "T0n": "heuristic_nfa", "T1": "dtw"}

RADII = [0.25, 0.50, 0.75, 1.00]
MARGINS = [0.10, 0.20, 0.30]
TOLERANCE = 0.25


def log(message: str) -> None:
    print(message, flush=True)


def load_words(file_id: str, mode: str) -> tuple[list[Word], dict]:
    payload = json.loads((RAW / f"{file_id}--{mode}.json").read_text(encoding="utf-8"))
    words = [Word(word["start"], word["end"], word["text"]) for word in payload["words"]]
    return words, payload


def diarization_segments(file_id: str, *, language: str) -> list[Segment]:
    """Reuse the diarization benchmark RTTMs (TitaNet @0.80) for VoxConverse; run the
    production diarization service for the real meeting and cache it."""
    if file_id != REAL_MEETING:
        candidates = list(DIAR_RESULTS.glob(f"scoring/*--d2-titanet--0.80--{file_id}/*.rttm"))
        if not candidates:
            raise SystemExit(f"diarization RTTM missing for {file_id}")
        return parse_rttm(candidates[0])

    cache = RAW / "real--diarization.json"
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        settings = Settings(diarization_threshold=0.80, diarization_threads=8)
        result = diarize(
            MEETINGS / file_id / "processing.wav", settings, requested_speaker_count=None
        )
        payload = {
            "num_speakers": result.num_speakers,
            "segments": [
                {"start": segment.start, "end": segment.end, "speaker": segment.speaker}
                for segment in result.segments
            ],
        }
        cache.write_text(json.dumps(payload), encoding="utf-8")
    return [
        Segment(segment["start"], segment["end"], segment["speaker"])
        for segment in payload["segments"]
    ]


def baseline_speakers(words: list[Word], segments: list[Segment]) -> list[str | None]:
    production = [DiarizationSegment(segment.start, segment.end, segment.speaker) for segment in segments]
    return [assign_speaker(word, production, TOLERANCE) for word in words]


def resolve(words: list[Word], baseline: list[str | None], segments: list[Segment], radius: float, margin: float):
    return apply_resolver(words, baseline, segments, radius=radius, margin=margin)


def evaluate(words, speakers, reference, baseline=None) -> dict:
    return {
        **evaluate_assignment(
            words,
            [f"speaker_{speaker}" if speaker is not None else None for speaker in speakers],
            reference,
            baseline_speakers=[
                f"speaker_{speaker}" if speaker is not None else None for speaker in (baseline or [])
            ]
            if baseline
            else None,
        ),
    }


def evaluate_real(words, speakers, baseline=None) -> dict:
    metrics = speaker_metrics(words, speakers)
    metrics.pop("kişi_count", None)
    return {
        "words": len(words),
        "assigned_words": sum(1 for speaker in speakers if speaker is not None),
        "unresolved_words": sum(1 for speaker in speakers if speaker is None),
        "distinct_speakers": len({speaker for speaker in speakers if speaker is not None}),
        **metrics,
        "newly_resolved_words": (
            sum(1 for index, s in enumerate(speakers) if s is not None and baseline[index] is None)
            if baseline is not None
            else 0
        ),
    }


def unresolved_causes(words, speakers, segments) -> dict:
    causes = {"overlap_region": 0, "coverage_gap": 0}
    for word, speaker in zip(words, speakers, strict=True):
        if speaker is not None:
            continue
        midpoint = (word.start + word.end) / 2.0
        owners = [segment for segment in segments if segment.start <= midpoint <= segment.end]
        if len(owners) >= 2:
            causes["overlap_region"] += 1
        else:
            causes["coverage_gap"] += 1
    return causes


def write_private_transcript(name: str, words, speakers) -> None:
    labels = label_speakers([f"speaker_{s}" for s in speakers if s is not None])
    lines = []
    for word, speaker in zip(words, speakers, strict=True):
        label = labels.get(f"speaker_{speaker}", "Bilinmeyen") if speaker is not None else "Bilinmeyen"
        if lines and lines[-1][0] == label:
            lines[-1][1] = f"{lines[-1][1]} {word.text}"
            lines[-1][2] = word.end
        else:
            lines.append([label, word.text, word.start, word.start])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / f"{name}.txt").write_text(
        "\n".join(f"[{entry[2]:8.3f}] {entry[0]}: {entry[1]}" for entry in lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    PRIVATE.mkdir(parents=True, exist_ok=True)
    aggregate: dict = {
        "dtw_check": {},
        "turkish_wer": {},
        "calibration": {},
        "selection": {},
        "validation": {},
        "real_meeting": {},
    }

    # --- DTW facts on the real meeting -------------------------------------
    for mode in ("heuristic", "heuristic_nfa", "dtw"):
        _, payload = load_words(REAL_MEETING, mode)
        aggregate["dtw_check"][mode] = {
            "words": payload["word_count"],
            "decoding_seconds": payload["decoding_seconds"],
            "flash_attention_disabled": payload.get("flash_attention_disabled", False),
        }
    heuristic_words, heuristic_payload = load_words(REAL_MEETING, "heuristic")
    dtw_words, dtw_payload = load_words(REAL_MEETING, "dtw")
    nfa_words, _ = load_words(REAL_MEETING, "heuristic_nfa")
    aggregate["dtw_check"]["text_identical_heuristic_vs_dtw"] = (
        heuristic_payload["text"] == dtw_payload["text"]
    )
    nfa_payload = json.loads((RAW / f"{REAL_MEETING}--heuristic_nfa.json").read_text(encoding="utf-8"))
    aggregate["dtw_check"]["text_identical_heuristic_vs_heuristic_nfa"] = (
        heuristic_payload["text"] == nfa_payload["text"]
    )
    aggregate["dtw_check"]["text_identical_heuristic_nfa_vs_dtw"] = (
        nfa_payload["text"] == dtw_payload["text"]
    )
    dtw_raw_words = dtw_payload["words"]
    shifts = sorted(
        abs(word["dtw_start"] - word["offset_start"])
        for word in dtw_raw_words
        if word.get("dtw_start") is not None
    )
    aggregate["dtw_check"]["timestamp_shift_seconds"] = {
        "median_abs_start_delta_vs_heuristic_offsets": round(shifts[len(shifts) // 2], 3) if shifts else None,
        "max_abs_start_delta_vs_heuristic_offsets": round(shifts[-1], 3) if shifts else None,
        "point_words": sum(1 for word in dtw_words if abs(word.end - word.start) < 1e-9),
    }

    # --- calibration grid --------------------------------------------------
    log("calibration grid on VoxConverse calibration split")
    per_combination: dict[str, dict] = {}
    for radius in RADII:
        for margin in MARGINS:
            totals = {"compared": 0, "matching": 0, "wrong": 0, "unresolved": 0, "words": 0}
            newly_resolved = 0
            for file_id in CALIBRATION:
                words, _ = load_words(file_id, "dtw")
                segments = diarization_segments(file_id, language="en")
                reference = parse_rttm(VOXCONVERSE / "voxconverse" / "dev" / f"{file_id}.rttm")
                baseline = baseline_speakers(words, segments)
                resolved, _ = resolve(words, baseline, segments, radius, margin)
                evaluation = evaluate(words, resolved, reference, baseline)
                totals["compared"] += evaluation["compared_words"]
                totals["matching"] += evaluation["matching_words"]
                totals["wrong"] += evaluation["wrong_attribution_words"]
                totals["unresolved"] += evaluation["unresolved_words"]
                totals["words"] += evaluation["words"]
                newly_resolved += evaluation["newly_resolved_words"]
            key = f"radius={radius:.2f},margin={margin:.2f}"
            per_combination[key] = {
                "radius": radius,
                "margin": margin,
                "agreement": round(totals["matching"] / totals["compared"], 4) if totals["compared"] else None,
                "wrong_attribution_rate": round(totals["wrong"] / totals["compared"], 4)
                if totals["compared"]
                else None,
                "unresolved_rate": round(totals["unresolved"] / totals["words"], 4),
                "newly_resolved_words": newly_resolved,
                "compared_words": totals["compared"],
            }
    aggregate["calibration"] = per_combination

    ranking = sorted(
        per_combination.items(),
        key=lambda item: (
            item[1]["wrong_attribution_rate"],
            -(item[1]["agreement"] or 0.0),
            item[1]["unresolved_rate"],
            item[1]["radius"],
            -item[1]["margin"],
        ),
    )
    selected_key, selected = ranking[0]
    aggregate["selection"] = {"key": selected_key, "config": selected}
    log(f"selected M2 config: {selected_key}")

    # --- held-out validation ----------------------------------------------
    log("held-out validation")
    for label, mode, use_resolver in (
        ("M0-heuristic", "heuristic", False),
        ("M0n-heuristic-nfa", "heuristic_nfa", False),
        ("M1-dtw", "dtw", False),
        ("M2-dtw-context", "dtw", True),
    ):
        totals = {"compared": 0, "matching": 0, "wrong": 0, "unresolved": 0, "words": 0}
        turns = 0
        rapid = 0
        stt_words = 0
        for file_id in VALIDATION:
            words, payload = load_words(file_id, mode)
            stt_words += payload["word_count"]
            segments = diarization_segments(file_id, language="en")
            reference = parse_rttm(VOXCONVERSE / "voxconverse" / "dev" / f"{file_id}.rttm")
            baseline = baseline_speakers(words, segments)
            speakers = (
                resolve(words, baseline, segments, selected["radius"], selected["margin"])[0]
                if use_resolver
                else baseline
            )
            evaluation = evaluate(words, speakers, reference, baseline)
            totals["compared"] += evaluation["compared_words"]
            totals["matching"] += evaluation["matching_words"]
            totals["wrong"] += evaluation["wrong_attribution_words"]
            totals["unresolved"] += evaluation["unresolved_words"]
            totals["words"] += evaluation["words"]
            metrics = speaker_metrics(words, speakers)
            turns += metrics["turns"]
            rapid += metrics["rapid_flips"]
        aggregate["validation"][label] = {
            "total_stt_words": stt_words,
            "agreement": round(totals["matching"] / totals["compared"], 4) if totals["compared"] else None,
            "wrong_attribution_rate": round(totals["wrong"] / totals["compared"], 4)
            if totals["compared"]
            else None,
            "unresolved_words": totals["unresolved"],
            "unresolved_rate": round(totals["unresolved"] / totals["words"], 4),
            "speaker_turns": turns,
            "rapid_flips": rapid,
        }

    # --- controlled Turkish WER (T0 / T0n / T1) ---------------------------
    log("controlled Turkish WER")
    from normalize import score as wer_score

    reference_text = Path("/stt-bench/reference_tr.txt").read_text(encoding="utf-8")
    turkish: dict = {"reference_words_per_recording": len(reference_text.split()), "modes": {}}
    for label, mode in WER_MODES.items():
        per_recording = {}
        totals = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
        for recording in TURKISH_CONTROLLED:
            words, payload = load_words(recording, mode)
            result = wer_score(reference_text, payload["text"])
            point_words = sum(1 for word in words if abs(word.end - word.start) < 1e-9)
            monotonicity = sum(
                1
                for index in range(1, len(words))
                if words[index].start < words[index - 1].end - 1e-9
            )
            per_recording[recording] = {
                "wer": round(result["wer"], 4),
                "substitutions": result["substitutions"],
                "deletions": result["deletions"],
                "insertions": result["insertions"],
                "word_count": payload["word_count"],
                "rtf": round(payload["decoding_seconds"] / payload["audio_seconds"], 4),
                "point_words": point_words,
                "zero_duration_words": point_words,
                "monotonicity_errors": monotonicity,
            }
            for key in totals:
                totals[key] += result[key]
        combined_errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
        turkish["modes"][label] = {
            "timestamp_mode": mode,
            "per_recording": per_recording,
            "combined": {
                **totals,
                "total_errors": combined_errors,
                "wer": round(combined_errors / totals["reference_words"], 4),
            },
        }
    turkish["text_identical_t0n_vs_t1"] = {
        recording: load_words(recording, "heuristic_nfa")[1]["text"]
        == load_words(recording, "dtw")[1]["text"]
        for recording in TURKISH_CONTROLLED
    }
    aggregate["turkish_wer"] = turkish

    # --- real four-speaker meeting ----------------------------------------
    log("real four-speaker meeting")
    segments = diarization_segments(REAL_MEETING, language="tr")
    for label, words, use_resolver in (
        ("M0-baseline", heuristic_words, False),
        ("M1-dtw", dtw_words, False),
        ("M2-context", dtw_words, True),
    ):
        baseline = baseline_speakers(words, segments)
        speakers = (
            resolve(words, baseline, segments, selected["radius"], selected["margin"])[0]
            if use_resolver
            else baseline
        )
        metrics = evaluate_real(words, speakers, baseline)
        metrics["unresolved_causes"] = unresolved_causes(words, speakers, segments)
        aggregate["real_meeting"][label] = metrics
        write_private_transcript(f"real-{label}", words, speakers)

    # Private attribution diff: only words whose speaker label changed between M0/M1/M2.
    diff_lines = []
    segment_speakers = {}
    for label, words, use_resolver in (
        ("M0", heuristic_words, False),
        ("M1", dtw_words, False),
        ("M2", dtw_words, True),
    ):
        baseline = baseline_speakers(words, segments)
        speakers = (
            resolve(words, baseline, segments, selected["radius"], selected["margin"])[0]
            if use_resolver
            else baseline
        )
        segment_speakers[label] = (words, speakers)

    m0_words, m0_speakers = segment_speakers["M0"]
    m1_words, m1_speakers = segment_speakers["M1"]
    m2_words, m2_speakers = segment_speakers["M2"]
    diff_lines.append(f"M0 words: {len(m0_words)} | M1/M2 words: {len(m1_words)}")
    diff_lines.append("")
    diff_lines.append("=== M0 -> M1 (same word index on the M1 word list where comparable) ===")
    for index, word in enumerate(m1_words):
        m0_label = m0_speakers[index] if index < len(m0_speakers) else None
        m1_label = m1_speakers[index]
        if m0_label != m1_label:
            diff_lines.append(
                f"  [{word.start:7.2f}] {word.text!r}: M0={m0_label or 'Bilinmeyen'} -> M1={m1_label or 'Bilinmeyen'}"
            )
    diff_lines.append("")
    diff_lines.append("=== M1 -> M2 (same words, resolver applied) ===")
    for index, word in enumerate(m2_words):
        m1_label = m1_speakers[index]
        m2_label = m2_speakers[index]
        if m1_label != m2_label:
            diff_lines.append(
                f"  [{word.start:7.2f}] {word.text!r}: M1={m1_label or 'Bilinmeyen'} -> M2={m2_label or 'Bilinmeyen'}"
            )
    (PRIVATE / "attribution-diff-M0-M1-M2.txt").write_text(
        "\n".join(diff_lines) + "\n", encoding="utf-8"
    )

    (HARNESS / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (PRIVATE / "aggregate-full.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"wrote {HARNESS / 'aggregate.json'} (metrics only) and private outputs in {PRIVATE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
