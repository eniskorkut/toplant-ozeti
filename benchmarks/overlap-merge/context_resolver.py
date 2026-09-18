"""Benchmark-only conservative context resolver (candidate merge M2).

The production baseline assignment stays authoritative. This module only proposes
resolutions for words the baseline left unresolved, with two conservative rules:

CASE A (ambiguous overlapping diarization): look at a local context window around the
word midpoint and sum how long each speaker is active inside it. Resolve only when one
speaker has the unique highest support AND beats the runner-up by a minimum margin.

CASE B (coverage gap): bridge only when the previous and next resolved words share the
same speaker, the unresolved word lies between them, and the nearest diarization
activity is within a small fixed distance. No long-range guessing, no "short words
inherit a neighbour" behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_TOLERANCE = 0.25
BRIDGE_MAX_GAP_SECONDS = 0.5


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str


def speaker_support(segments: list[Segment], window_start: float, window_end: float) -> dict[str, float]:
    """Active duration per speaker inside [window_start, window_end]."""
    support: dict[str, float] = {}
    for segment in segments:
        overlap = min(window_end, segment.end) - max(window_start, segment.start)
        if overlap > 0:
            support[segment.speaker] = support.get(segment.speaker, 0.0) + overlap
    return support


def nearest_activity_distance(word_start: float, word_end: float, segments: list[Segment]) -> float:
    """Distance from the word interval to the closest diarization activity."""
    if not segments:
        return float("inf")
    return min(
        max(0.0, max(segment.start - word_end, word_start - segment.end))
        for segment in segments
    )


def resolve_contextual(
    words: list,
    speakers: list[str | None],
    segments: list[Segment],
    *,
    radius: float,
    margin: float,
) -> tuple[list[str | None], dict]:
    """CASE A only: resolve unresolved words by local speaker support."""
    resolved = list(speakers)
    stats = {"considered": 0, "resolved": 0, "rejected_margin": 0, "rejected_tie": 0}

    for index, speaker in enumerate(speakers):
        if speaker is not None:
            continue
        stats["considered"] += 1
        word = words[index]
        midpoint = (word.start + word.end) / 2.0
        support = speaker_support(segments, midpoint - radius, midpoint + radius)
        if not support:
            continue
        ranked = sorted(support.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) >= 2 and ranked[1][1] == ranked[0][1]:
            stats["rejected_tie"] += 1
            continue
        if len(ranked) >= 2 and (ranked[0][1] - ranked[1][1]) < margin:
            stats["rejected_margin"] += 1
            continue
        resolved[index] = ranked[0][0]
        stats["resolved"] += 1

    return resolved, stats


def bridge_coverage_gaps(
    words: list,
    speakers: list[str | None],
    segments: list[Segment],
    *,
    max_gap: float = BRIDGE_MAX_GAP_SECONDS,
) -> tuple[list[str | None], dict]:
    """CASE B only: bridge across small coverage gaps between equal-speaker words."""
    resolved = list(speakers)
    stats = {"considered": 0, "resolved": 0}

    for index, speaker in enumerate(speakers):
        if speaker is not None:
            continue
        previous = next(
            (speakers[position] for position in range(index - 1, -1, -1) if speakers[position]),
            None,
        )
        following = next(
            (speakers[position] for position in range(index + 1, len(speakers)) if speakers[position]),
            None,
        )
        if previous is None or following is None or previous != following:
            continue
        stats["considered"] += 1
        word = words[index]
        if nearest_activity_distance(word.start, word.end, segments) <= max_gap:
            resolved[index] = previous
            stats["resolved"] += 1

    return resolved, stats


def apply_resolver(
    words: list,
    baseline_speakers: list[str | None],
    segments: list[Segment],
    *,
    radius: float,
    margin: float,
    bridge_max_gap: float = BRIDGE_MAX_GAP_SECONDS,
) -> tuple[list[str | None], dict]:
    """CASE A first, then CASE B on whatever is still unresolved."""
    after_context, context_stats = resolve_contextual(
        words, baseline_speakers, segments, radius=radius, margin=margin
    )
    after_bridge, bridge_stats = bridge_coverage_gaps(
        words, after_context, segments, max_gap=bridge_max_gap
    )
    return after_bridge, {
        "baseline_unresolved": sum(1 for speaker in baseline_speakers if speaker is None),
        "resolved_by_context": context_stats["resolved"],
        "resolved_by_bridge": bridge_stats["resolved"],
        "remaining_unresolved": sum(1 for speaker in after_bridge if speaker is None),
        "context": context_stats,
        "bridge": bridge_stats,
    }
