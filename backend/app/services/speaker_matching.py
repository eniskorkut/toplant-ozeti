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
# A mapping must beat the runner-up clearly, otherwise it stays provisional.
MIN_MARGIN_RATIO = 0.2
MIN_MARGIN_SECONDS = 0.3
# Gap continuity must also be clearly nearest: two canonicals at a similar
# distance from a new segment mean we cannot tell them apart.
GAP_MARGIN_SECONDS = 0.5
GAP_MARGIN_RATIO = 1.5
# Provider splits of one person are either overlapping sub-segments or almost
# continuous speech. Anything else (a short turn after a real pause) may be the
# OTHER person: it must stay a candidate instead of being absorbed.
MERGE_GAP_SECONDS = 0.3
FRAGMENT_MAX_SECONDS = 1.0
# Claiming an unclaimed canonical across a pause: measured against the full-file
# reference, the wider tolerance kept stability and agreement better than a tight
# one (the long snapshots provide overlap evidence; this only covers resumptions).
CONTINUATION_GAP_SECONDS = 2.0

Interval = tuple[float, float]


@dataclass(frozen=True)
class SpeakerMatch:
    provider_speaker: str
    canonical_speaker: str
    overlap_seconds: float
    ratio: float
    evidence: str = "overlap"
    # Best-vs-second-best overlap margin in seconds (0 for non-overlap evidence).
    margin: float = 0.0


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
    min_margin_ratio: float = MIN_MARGIN_RATIO,
    min_margin_seconds: float = MIN_MARGIN_SECONDS,
) -> tuple[dict[str, SpeakerMatch], list[str], list[str]]:
    """Match provider speakers to canonical speakers with confidence filtering.

    Primary evidence is temporal overlap inside the shared region. A candidate is
    only accepted when it beats the runner-up by a clear margin; otherwise the
    provider speaker is reported as **ambiguous** (provisional) instead of being
    forced onto a canonical speaker. When `gap_tolerance_seconds` is set (live
    tracking only), speakers with no overlap may still be linked across a short
    conversational pause, and `allow_disjoint_merge` folds request-local splits of
    one person into a single canonical speaker.

    Returns (accepted matches, unmatched, ambiguous). Unmatched speakers have no
    evidence at all (candidates for a new canonical); ambiguous speakers have
    evidence but it is not decisive (they must stay provisional).
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

    # Margin gate: prune dominated edges and remember speakers whose best edge is
    # too close to the second best to be trustworthy.
    ambiguous: set[str] = set()
    if min_margin_ratio > 0 or min_margin_seconds > 0:
        best_by_provider: dict[str, list[tuple[float, str]]] = {}
        for (canonical_name, provider_name), score in accepted.items():
            best_by_provider.setdefault(provider_name, []).append((score, canonical_name))
        pruned: dict[tuple[str, str], float] = {}
        for provider_name, edges in best_by_provider.items():
            edges.sort(key=lambda item: (-item[0], item[1]))
            best_score, best_label = edges[0]
            second_score = edges[1][0] if len(edges) > 1 else 0.0
            required = max(min_margin_seconds, best_score * min_margin_ratio)
            if second_score > 0 and best_score - second_score < required:
                ambiguous.add(provider_name)
                continue
            pruned[(best_label, provider_name)] = best_score
        accepted = pruned

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
        pending = [
            name
            for name in provider_intervals
            if name not in matching and name not in ambiguous
        ]
        while pending:
            best: tuple[float, str, str] | None = None
            second_best: dict[str, float] = {}
            for provider_name in pending:
                provider_spans = provider_intervals[provider_name]
                gaps: list[tuple[float, str]] = []
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
                        gaps.append((gap, canonical_name))
                if not gaps:
                    continue
                gaps.sort(key=lambda item: (item[0], item[1]))
                second_best[provider_name] = gaps[1][0] if len(gaps) > 1 else float("inf")
                candidate = (gaps[0][0], gaps[0][1], provider_name)
                if best is None or candidate < best:
                    best = candidate
            if best is None:
                break
            gap, canonical_name, provider_name = best
            if second_best.get(provider_name, float("inf")) <= max(
                gap * GAP_MARGIN_RATIO, gap + GAP_MARGIN_SECONDS
            ):
                # Two canonicals are equally plausible continuations: stay
                # provisional instead of merging into the wrong person.
                ambiguous.add(provider_name)
                pending.remove(provider_name)
                continue
            existing = assigned.get(canonical_name)
            if existing is None:
                if gap > CONTINUATION_GAP_SECONDS:
                    # Not a continuation: leave the cluster as promotion evidence.
                    pending.remove(provider_name)
                    continue
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
                or (
                    # A true provider split shows overlapping sub-segments of the
                    # same voice; short duration alone is not evidence.
                    not intervals_disjoint(
                        provider_intervals[provider_name], provider_intervals[existing]
                    )
                    and total_seconds(provider_intervals[provider_name]) <= FRAGMENT_MAX_SECONDS
                )
            ):
                matching[provider_name] = canonical_name
                effective[canonical_name] = effective.get(canonical_name, []) + list(
                    provider_intervals[provider_name]
                )
                evidence[provider_name] = "merged-fragment"
                pending.remove(provider_name)
            else:
                pending.remove(provider_name)

    # Margin per provider speaker: best accepted score minus the runner-up, used
    # by conservative confirmation policies (never for matching itself).
    margin_by_provider: dict[str, float] = {}
    for provider_name in provider_intervals:
        provider_scores = sorted(
            (
                score
                for (canonical_name, other), score in scores.items()
                if other == provider_name
            ),
            reverse=True,
        )
        margin_by_provider[provider_name] = round(
            provider_scores[0] - provider_scores[1], 3
        ) if len(provider_scores) > 1 else round(provider_scores[0], 3) if provider_scores else 0.0

    matches: dict[str, SpeakerMatch] = {}
    for provider_name, canonical_name in matching.items():
        score = scores.get((canonical_name, provider_name), 0.0)
        matches[provider_name] = SpeakerMatch(
            provider_speaker=provider_name,
            canonical_speaker=canonical_name,
            overlap_seconds=round(score, 3),
            ratio=round(score / provider_totals[provider_name], 3),
            evidence=evidence[provider_name],
            margin=margin_by_provider.get(provider_name, 0.0),
        )
    unmatched = [
        name
        for name in provider_intervals
        if name not in matches and name not in ambiguous
    ]
    return matches, sorted(unmatched), sorted(ambiguous)


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
    """Whole-meeting mapping of final provider speakers onto dense canonical labels.

    1. Matches final provider speakers against confirmed live canonical speakers
       via maximum-weight bipartite matching with confidence and margin gating.
    2. Guarantees the final canonical speaker set is strictly dense:
       Kişi 1, Kişi 2, ..., Kişi M (where M = len(provider_intervals)).
    3. Confident matches preserve their live canonical label (and alias) whenever
       within the dense target pool.
    4. Unmatched or ambiguous final speakers receive the next unused Kişi label
       in order of first appearance.
    5. Discards any abandoned or out-of-range provisional identities (no gaps,
       never Kişi 5 when M=2).
    """
    total_final = len(provider_intervals)
    if total_final == 0:
        return {}

    matches, unmatched, ambiguous = match_speakers(
        canonical_intervals, provider_intervals
    )
    unmatched_set = set(unmatched) | set(ambiguous)

    # Final provider speakers ordered by first speech appearance in audio
    provider_by_appearance = sorted(
        provider_intervals.keys(),
        key=lambda p: min((start for start, _ in provider_intervals[p]), default=0.0),
    )

    dense_target_pool = [f"Kişi {i}" for i in range(1, total_final + 1)]
    assigned_mapping: dict[str, str] = {}
    used_labels: set[str] = set()

    # Step 1: Assign confident matches if the matched label is in the dense pool
    # and not already taken. Strongest overlap evidence wins.
    sorted_matches = sorted(
        matches.items(),
        key=lambda item: item[1].overlap_seconds,
        reverse=True,
    )
    for provider_name, match in sorted_matches:
        cand_label = match.canonical_speaker
        if cand_label in dense_target_pool and cand_label not in used_labels:
            assigned_mapping[provider_name] = cand_label
            used_labels.add(cand_label)
        else:
            unmatched_set.add(provider_name)

    # Step 2: For all remaining provider speakers (in order of first appearance),
    # allocate the next available label from dense_target_pool.
    remaining_providers = [p for p in provider_by_appearance if p not in assigned_mapping]
    available_labels = [label for label in dense_target_pool if label not in used_labels]

    for p, label in zip(remaining_providers, available_labels, strict=True):
        assigned_mapping[p] = label
        used_labels.add(label)

    return assigned_mapping


def reconcile_final_aliases(
    canonical_intervals: dict[str, list[Interval]],
    provider_intervals: dict[str, list[Interval]],
    live_aliases: dict[str, str],
) -> dict[str, str]:
    """Preserve aliases only when the live canonical speaker confidently mapped to a final speaker.

    Ambiguous or unmatched speakers never inherit an alias. Prefer losing an
    uncertain alias over misidentifying a real person.
    """
    matches, unmatched, ambiguous = match_speakers(
        canonical_intervals, provider_intervals
    )
    mapping = remap_final_speakers(canonical_intervals, provider_intervals)

    final_aliases: dict[str, str] = {}
    for provider_name, match in matches.items():
        live_label = match.canonical_speaker
        if live_label in live_aliases:
            final_label = mapping.get(provider_name)
            if final_label:
                final_aliases[final_label] = live_aliases[live_label]
    return final_aliases

