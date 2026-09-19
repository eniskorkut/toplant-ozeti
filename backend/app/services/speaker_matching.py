"""Cross-window speaker matching.

Provider speaker ids are request-local, so they are never compared by name. Two
speakers are considered the same person only through temporal evidence: the speech
overlap (in seconds) inside the region the two windows share. Matching is a
deterministic maximum-weight one-to-one assignment; pairs below a confidence
threshold are rejected and the caller allocates a new canonical speaker.

No embeddings, no voiceprints, no cross-meeting state: timestamps only.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_OVERLAP_SECONDS = 0.4
MIN_OVERLAP_RATIO = 0.3
DEFAULT_GAP_TOLERANCE_SECONDS = 2.0
# Splitting one person into several request-local labels happens at segment
# boundaries, so a merge requires a short silence.
MERGE_GAP_SECONDS = 0.75
# ...or a short "fragment" label that overlaps the longer label of the same person.
FRAGMENT_MAX_SECONDS = 1.0

Interval = tuple[float, float]


@dataclass(frozen=True)
class SpeakerMatch:
    provider_speaker: str
    canonical_speaker: str
    overlap_seconds: float
    ratio: float
    evidence: str = "overlap"


def interval_gap(left: Interval, right: Interval) -> float:
    """Temporal distance between two intervals; 0 when they touch or overlap."""
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0.0


def intervals_disjoint(left: list[Interval], right: list[Interval]) -> bool:
    """True when the two speakers never talked at the same time."""
    for left_span in left:
        for right_span in right:
            if overlap_seconds(left_span, right_span) > 0:
                return False
    return True


def overlap_seconds(left: Interval, right: Interval) -> float:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    return max(0.0, end - start)


def clip_intervals(intervals: list[Interval], window: Interval) -> list[Interval]:
    clipped: list[Interval] = []
    for start, end in intervals:
        span = overlap_seconds((start, end), window)
        if span > 0:
            clipped.append((max(start, window[0]), min(end, window[1])))
    return clipped


def total_seconds(intervals: list[Interval]) -> float:
    return sum(max(0.0, end - start) for start, end in intervals)


def score_pairs(
    canonical_intervals: dict[str, list[Interval]],
    provider_intervals: dict[str, list[Interval]],
    *,
    window: Interval | None = None,
) -> dict[tuple[str, str], float]:
    """Summed temporal overlap for every (canonical, provider) pair."""
    canonical = canonical_intervals
    provider = provider_intervals
    if window is not None:
        canonical = {name: clip_intervals(spans, window) for name, spans in canonical.items()}
        provider = {name: clip_intervals(spans, window) for name, spans in provider.items()}

    scores: dict[tuple[str, str], float] = {}
    for canonical_name, canonical_spans in canonical.items():
        for provider_name, provider_spans in provider.items():
            total = 0.0
            for left in canonical_spans:
                for right in provider_spans:
                    total += overlap_seconds(left, right)
            if total > 0:
                scores[(canonical_name, provider_name)] = total
    return scores


def maximum_weight_matching(
    scores: dict[tuple[str, str], float],
) -> dict[str, str]:
    """Deterministic maximum-weight one-to-one matching (provider -> canonical).

    Ties prefer the lower canonical index, then the lower provider order, so the
    result is stable across runs. Implemented with the Hungarian algorithm on a
    padded square cost matrix (sizes here are single-digit, so cost is irrelevant).
    """
    canonical_names = sorted({pair[0] for pair in scores})
    provider_names = sorted({pair[1] for pair in scores})
    if not canonical_names or not provider_names:
        return {}

    size = max(len(canonical_names), len(provider_names))
    scale = 100_000
    max_score = max(scores.values()) if scores else 0.0
    infinity = (int(max_score * scale) + 1) * (size + 1)

    # cost[i][j] for i in canonical, j in provider; padding rows/columns are free.
    cost: list[list[int]] = []
    for i in range(size):
        row: list[int] = []
        for j in range(size):
            if i < len(canonical_names) and j < len(provider_names):
                score = scores.get((canonical_names[i], provider_names[j]), 0.0)
                # Small deterministic tie-break: lower indices win.
                tie = i * size + j
                row.append(int((max_score - score) * scale) * (size * size + 1) + tie)
            else:
                row.append(0)
        cost.append(row)

    # Hungarian algorithm (minimization) over the square matrix.
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [infinity] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = infinity
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    matching: dict[str, str] = {}
    for j in range(1, size + 1):
        if p[j] == 0:
            continue
        i = p[j] - 1
        j_index = j - 1
        if i >= len(canonical_names) or j_index >= len(provider_names):
            continue
        provider_name = provider_names[j_index]
        canonical_name = canonical_names[i]
        if (canonical_name, provider_name) in scores:
            matching[provider_name] = canonical_name
    return matching


def match_speakers(
    canonical_intervals: dict[str, list[Interval]],
    provider_intervals: dict[str, list[Interval]],
    *,
    window: Interval | None = None,
    min_overlap_seconds: float = MIN_OVERLAP_SECONDS,
    min_overlap_ratio: float = MIN_OVERLAP_RATIO,
    gap_tolerance_seconds: float = 0.0,
    allow_disjoint_merge: bool = False,
) -> tuple[dict[str, SpeakerMatch], list[str]]:
    """Match provider speakers to canonical speakers with confidence filtering.

    Primary evidence is temporal overlap inside the shared region. When
    `gap_tolerance_seconds` is set (live tracking only), speakers with no overlap
    may still be linked to a canonical speaker when the silence between them is
    short — a conversational pause must not mint a new person. With
    `allow_disjoint_merge`, several request-local speakers that never talk at the
    same time may fold into one canonical speaker (the provider sometimes splits
    one person into multiple labels inside a single window).

    Returns (accepted matches by provider speaker, unmatched provider speakers).
    """
    scores = score_pairs(canonical_intervals, provider_intervals, window=window)
    provider_totals = {
        name: max(total_seconds(spans), 1e-6) for name, spans in provider_intervals.items()
    }

    accepted: dict[tuple[str, str], float] = {}
    for pair, score in scores.items():
        ratio = score / provider_totals[pair[1]]
        if score >= min_overlap_seconds and ratio >= min_overlap_ratio:
            accepted[pair] = score

    matching = maximum_weight_matching(accepted)
    evidence = {provider: "overlap" for provider in matching}

    if gap_tolerance_seconds > 0:
        assigned: dict[str, str] = {
            canonical_name: provider_name for provider_name, canonical_name in matching.items()
        }
        # Gaps are measured against everything assigned to a canonical so far,
        # including labels matched earlier in this same window.
        effective: dict[str, list[Interval]] = {
            label: list(spans) for label, spans in canonical_intervals.items()
        }
        pending = [name for name in provider_intervals if name not in matching]
        while pending:
            best: tuple[float, str, str] | None = None
            for provider_name in pending:
                provider_spans = provider_intervals[provider_name]
                for canonical_name, canonical_spans in effective.items():
                    gap = min(
                        (
                            interval_gap(provider_span, canonical_span)
                            for provider_span in provider_spans
                            for canonical_span in canonical_spans
                        ),
                        default=float("inf"),
                    )
                    if gap <= gap_tolerance_seconds:
                        candidate = (gap, canonical_name, provider_name)
                        if best is None or candidate < best:
                            best = candidate
            if best is None:
                break
            gap, canonical_name, provider_name = best
            existing = assigned.get(canonical_name)
            if existing is None:
                matching[provider_name] = canonical_name
                assigned[canonical_name] = provider_name
                effective[canonical_name] = effective.get(canonical_name, []) + list(
                    provider_intervals[provider_name]
                )
                evidence[provider_name] = "gap"
                pending.remove(provider_name)
            elif allow_disjoint_merge and (
                (
                    gap <= MERGE_GAP_SECONDS
                    and intervals_disjoint(
                        provider_intervals[provider_name], provider_intervals[existing]
                    )
                )
                or total_seconds(provider_intervals[provider_name]) <= FRAGMENT_MAX_SECONDS
            ):
                matching[provider_name] = canonical_name
                effective[canonical_name] = effective.get(canonical_name, []) + list(
                    provider_intervals[provider_name]
                )
                evidence[provider_name] = "merged-fragment"
                pending.remove(provider_name)
            else:
                pending.remove(provider_name)

    matches: dict[str, SpeakerMatch] = {}
    for provider_name, canonical_name in matching.items():
        score = scores.get((canonical_name, provider_name), 0.0)
        matches[provider_name] = SpeakerMatch(
            provider_speaker=provider_name,
            canonical_speaker=canonical_name,
            overlap_seconds=round(score, 3),
            ratio=round(score / provider_totals[provider_name], 3),
            evidence=evidence[provider_name],
        )
    unmatched = [name for name in provider_intervals if name not in matches]
    return matches, sorted(unmatched)


def next_canonical_label(existing: set[str]) -> str:
    """Kişi N with the smallest unused N (deterministic, meeting-local only)."""
    index = 1
    while f"Kişi {index}" in existing:
        index += 1
    return f"Kişi {index}"


def remap_final_speakers(
    canonical_intervals: dict[str, list[Interval]],
    provider_intervals: dict[str, list[Interval]],
) -> dict[str, str]:
    """Whole-meeting mapping of final provider speakers onto canonical live labels.

    Confident matches keep the live canonical label (and therefore any alias).
    Unmatched final speakers get fresh Kişi N labels: a possibly wrong alias is
    never applied to an unrelated speaker.
    """
    matches, unmatched = match_speakers(canonical_intervals, provider_intervals)
    mapping: dict[str, str] = {
        provider: match.canonical_speaker for provider, match in matches.items()
    }
    used = set(canonical_intervals) | set(mapping.values())
    for provider_name in unmatched:
        label = next_canonical_label(used)
        used.add(label)
        mapping[provider_name] = label
    return mapping
