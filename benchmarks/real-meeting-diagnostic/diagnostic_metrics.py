"""Pure, standard-library metric helpers for the real-meeting diagnostic.

Kept free of model code so the metric logic is unit-testable without loading any ML
runtime. Transcript text is only used to compute counts/ratios here; callers decide
what may be written to disk.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

UNKNOWN_LABEL = "Bilinmeyen"


def normalize_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, collapse whitespace (Turkish letters preserved)."""
    lowered = text.lower().replace("\u0307", "")
    cleaned = WORD_RE.sub(" ", lowered)
    return [word for word in WHITESPACE_RE.sub(" ", cleaned).split(" ") if word]


def timestamp_health(words: list) -> dict:
    """Structural timestamp diagnostics for STT word objects (start/end/text)."""
    zero_duration = 0
    monotonicity = 0
    out_of_range = 0
    previous_end = None
    for word in words:
        if abs(word.end - word.start) < 1e-9:
            zero_duration += 1
        if word.end < word.start:
            monotonicity += 1
        if previous_end is not None and word.start < previous_end - 1e-9:
            monotonicity += 1
        if word.start < -1e-9:
            out_of_range += 1
        previous_end = max(previous_end or 0.0, word.end)
    return {
        "words": len(words),
        "zero_duration_words": zero_duration,
        "monotonicity_violations": monotonicity,
        "negative_timestamps": out_of_range,
    }


def text_delta(auto_words: list[str], tr_words: list[str]) -> dict:
    """Factual difference diagnostics between two normalized word sequences."""
    matcher = difflib.SequenceMatcher(a=auto_words, b=tr_words, autojunk=False)
    differing = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        differing += max(i2 - i1, j2 - j1)
    return {
        "auto_words": len(auto_words),
        "tr_words": len(tr_words),
        "differing_word_positions": differing,
        "similarity_ratio": round(matcher.ratio(), 4),
    }


def short_examples(
    auto_words: list[str],
    tr_words: list[str],
    *,
    max_examples: int = 5,
    context: int = 6,
) -> list[dict]:
    """Up to `max_examples` short observational snippets where the two differ."""
    matcher = difflib.SequenceMatcher(a=auto_words, b=tr_words, autojunk=False)
    examples: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        auto_span = auto_words[max(0, i1 - context) : min(len(auto_words), i2 + context)]
        tr_span = tr_words[max(0, j1 - context) : min(len(tr_words), j2 + context)]
        examples.append(
            {
                "auto_snippet": " ".join(auto_span)[:160],
                "tr_snippet": " ".join(tr_span)[:160],
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def merge_metrics(turns: list, audio_seconds: float, *, real_speakers: int | None) -> dict:
    """Turn-level diagnostics from production SpeakerTurn objects."""
    kişi_labels = sorted(
        {turn.speaker for turn in turns if turn.speaker != UNKNOWN_LABEL},
        key=lambda label: int(label.split()[-1]) if label.split()[-1].isdigit() else 0,
    )
    unknown_turns = [turn for turn in turns if turn.speaker == UNKNOWN_LABEL]
    per_speaker_words: Counter = Counter()
    per_speaker_seconds: Counter = Counter()
    for turn in turns:
        per_speaker_words[turn.speaker] += turn.words
        per_speaker_seconds[turn.speaker] += max(0.0, turn.end - turn.start)

    speaker_changes = sum(
        1 for index in range(1, len(turns)) if turns[index].speaker != turns[index - 1].speaker
    )
    rapid_flips = 0
    for index in range(1, len(turns) - 1):
        previous, current, following = turns[index - 1], turns[index], turns[index + 1]
        if (
            current.speaker != previous.speaker
            and current.speaker != following.speaker
            and (current.end - current.start) < 0.4
        ):
            rapid_flips += 1

    merged_exchanges = [
        turn
        for turn in turns
        if (turn.end - turn.start) > 10.0 and turn.text.count(" - ") >= 2
    ]

    result = {
        "kişi_labels": kişi_labels,
        "final_speaker_count": len(kişi_labels),
        "turn_count": len(turns),
        "unknown_turn_count": len(unknown_turns),
        "unknown_seconds": round(sum(turn.end - turn.start for turn in unknown_turns), 3),
        "speaker_changes": speaker_changes,
        "rapid_flips": rapid_flips,
        "per_speaker_words": dict(sorted(per_speaker_words.items())),
        "per_speaker_seconds": {
            key: round(value, 2) for key, value in sorted(per_speaker_seconds.items())
        },
        "long_merged_exchange_turns": {
            "count": len(merged_exchanges),
            "durations": [round(turn.end - turn.start, 1) for turn in merged_exchanges],
        },
        "longest_turn_seconds": round(max((turn.end - turn.start for turn in turns), default=0.0), 2),
        "shortest_non_empty_turn_seconds": round(
            min(
                (turn.end - turn.start for turn in turns if turn.text.strip()),
                default=0.0,
            ),
            2,
        ),
    }
    if real_speakers is not None:
        result["speaker_count_error"] = len(kişi_labels) - real_speakers
    return result


def diarization_metrics(segments: list, audio_seconds: float) -> dict:
    """Cluster-level diagnostics from production diarization segments."""
    clusters: Counter = Counter()
    for segment in segments:
        clusters[segment.speaker] += max(0.0, segment.end - segment.start)
    per_cluster_segments: Counter = Counter()
    for segment in segments:
        per_cluster_segments[segment.speaker] += 1
    return {
        "non_empty_clusters": len([value for value in clusters.values() if value > 0]),
        "segment_count": len(segments),
        "per_cluster_seconds": {key: round(value, 2) for key, value in sorted(clusters.items())},
        "per_cluster_segments": dict(sorted(per_cluster_segments.items())),
    }


def segment_coverage(segments: list, audio_seconds: float) -> dict:
    """How much of the recording the diarization actually covers (union of segments)."""
    if not segments:
        return {"coverage_seconds": 0.0, "coverage_ratio": 0.0, "longest_gap_seconds": audio_seconds}
    intervals = sorted((segment.start, segment.end) for segment in segments)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    covered = sum(end - start for start, end in merged)
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    gaps.append(merged[0][0])
    gaps.append(max(0.0, audio_seconds - merged[-1][1]))
    return {
        "coverage_seconds": round(covered, 3),
        "coverage_ratio": round(covered / audio_seconds, 4) if audio_seconds else 0.0,
        "longest_gap_seconds": round(max(gaps) if gaps else 0.0, 3),
    }


def unresolved_word_gaps(unresolved_words: list, segments: list) -> dict:
    """Distance from each unresolved word to the nearest diarization segment.

    Small gaps (a few hundred ms) point at the merge tolerance / coarse STT
    timestamps; large gaps point at diarization coverage (missed speech).
    """
    gaps: list[float] = []
    for word in unresolved_words:
        if not segments:
            gaps.append(float("inf"))
            continue
        gaps.append(
            max(
                0.0,
                min(
                    max(segment.start - word.end, word.start - segment.end)
                    for segment in segments
                ),
            )
        )
    if not gaps:
        return {"count": 0, "median_gap_seconds": None, "max_gap_seconds": None, "buckets": {}}
    ordered = sorted(gaps)
    return {
        "count": len(gaps),
        "median_gap_seconds": round(ordered[len(ordered) // 2], 3),
        "max_gap_seconds": None if ordered[-1] == float("inf") else round(ordered[-1], 3),
        "buckets": {
            "overlapping_or_within_tolerance": sum(1 for gap in gaps if gap <= 0.25),
            "tolerance_to_1s": sum(1 for gap in gaps if 0.25 < gap <= 1.0),
            "one_to_3s": sum(1 for gap in gaps if 1.0 < gap <= 3.0),
            "over_3s": sum(1 for gap in gaps if gap > 3.0),
        },
    }


def unresolved_causes(unresolved_words: list, segments: list, tolerance: float = 0.25) -> dict:
    """Classify WHY each unresolved word stayed unresolved.

    - overlap_region: two or more diarization segments contain the word midpoint,
      i.e. the speakers overlap there and no single speaker owns the midpoint;
    - coverage_gap: the word is farther than the tolerance from every segment
      (diarization reported no speech there);
    - boundary_or_tie: the word touches a single segment but could not be resolved
      (touching boundary / degenerate interval), which is a merge-repair candidate.
    """
    causes = {"overlap_region": 0, "coverage_gap": 0, "boundary_or_tie": 0}
    for word in unresolved_words:
        midpoint = (word.start + word.end) / 2.0
        owners = [segment for segment in segments if segment.start <= midpoint <= segment.end]
        if len(owners) >= 2:
            causes["overlap_region"] += 1
            continue
        nearest = min(
            (
                max(0.0, max(segment.start - word.end, word.start - segment.end))
                for segment in segments
            ),
            default=float("inf"),
        )
        if nearest > tolerance:
            causes["coverage_gap"] += 1
        else:
            causes["boundary_or_tie"] += 1
    total = sum(causes.values())
    return {
        "total": total,
        **causes,
        "shares": {
            key: round(value / total, 3) if total else 0.0 for key, value in causes.items()
        },
    }
