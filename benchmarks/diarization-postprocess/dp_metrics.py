"""Pure metrics for the diarization post-processing sweep (standard library only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str


def _merged_intervals(segments: list[Segment]) -> list[list[float]]:
    intervals = sorted((segment.start, segment.end) for segment in segments)
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def coverage_metrics(segments: list[Segment], audio_seconds: float) -> dict:
    merged = _merged_intervals(segments)
    covered = sum(end - start for start, end in merged)
    gaps: list[float] = []
    if merged:
        gaps.append(merged[0][0])
        gaps.extend(merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1))
        gaps.append(max(0.0, audio_seconds - merged[-1][1]))
    else:
        gaps.append(audio_seconds)
    gaps = [gap for gap in gaps if gap > 1e-9]
    return {
        "coverage_seconds": round(covered, 3),
        "coverage_ratio": round(covered / audio_seconds, 4) if audio_seconds else 0.0,
        "uncovered_gap_seconds": round(sum(gaps), 3),
        "longest_uncovered_gap_seconds": round(max(gaps), 3) if gaps else 0.0,
    }


def overlap_metrics(segments: list[Segment], audio_seconds: float) -> dict:
    """Total time where two or more clusters are simultaneously active."""
    boundaries = sorted({point for segment in segments for point in (segment.start, segment.end)})
    overlap_seconds = 0.0
    for index in range(len(boundaries) - 1):
        left, right = boundaries[index], boundaries[index + 1]
        if right <= left:
            continue
        active = sum(1 for segment in segments if segment.start <= left and segment.end >= right)
        if active >= 2:
            overlap_seconds += right - left
    return {
        "overlap_seconds": round(overlap_seconds, 3),
        "overlap_ratio": round(overlap_seconds / audio_seconds, 4) if audio_seconds else 0.0,
    }


def segment_duration_buckets(segments: list[Segment]) -> dict:
    buckets = {"lt_0.3s": 0, "0.3_to_0.5s": 0, "0.5_to_1.0s": 0, "gt_1.0s": 0}
    for segment in segments:
        duration = segment.end - segment.start
        if duration < 0.3:
            buckets["lt_0.3s"] += 1
        elif duration < 0.5:
            buckets["0.3_to_0.5s"] += 1
        elif duration < 1.0:
            buckets["0.5_to_1.0s"] += 1
        else:
            buckets["gt_1.0s"] += 1
    return buckets


def unresolved_causes(words: list, speakers: list, segments: list[Segment]) -> dict:
    causes = {"overlap_ambiguity": 0, "coverage_gap": 0, "other": 0}
    for word, speaker in zip(words, speakers, strict=True):
        if speaker is not None:
            continue
        midpoint = (word.start + word.end) / 2.0
        owners = [segment for segment in segments if segment.start <= midpoint <= segment.end]
        if len(owners) >= 2:
            causes["overlap_ambiguity"] += 1
            continue
        nearest = min(
            (
                max(0.0, max(segment.start - word.end, word.start - segment.end))
                for segment in segments
            ),
            default=float("inf"),
        )
        if nearest > 0.25:
            causes["coverage_gap"] += 1
        else:
            causes["other"] += 1
    return causes
