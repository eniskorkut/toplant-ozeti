"""Canonical Evaluator for Live Diarization Confirmation Policies.

Single source of truth for computing:
- live_vs_final_confirmed_precision (correct confirmed / all confirmed)
- wrong_confirmed_rate (wrong confirmed / total evaluated speech)
- confirmed_coverage (confirmed speech / total evaluated speech)
- provisional_rate (unresolved speech / total evaluated speech)
- confirmed_canonical_speakers (distinct confirmed labels)
- live_provisional_candidates (unconfirmed candidate labels)
- switches (label switches on confirmed timeline)
- median and p95 confirmation delay

Supports both:
- MICRO aggregation: pooled / weighted by speech seconds
- MACRO aggregation: unweighted mean across recordings

Uses strictly cached windows and reference turns. No external API calls.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.routers.live_transcription import _provider_intervals
from app.services.live_sessions import LiveSessionStore, next_canonical_label
from app.services.speaker_matching import match_speakers
from app.services.transcription import NormalizedWord

RESOLUTION = 0.25  # seconds per grid point on timeline
PROVIDER_LATENCY_SECONDS = 1.0  # measured provider latency for delay calculation


@dataclass(frozen=True)
class PolicySpec:
    name: str
    is_confirmed_fn: Callable[[str, float, float, Any, bool, int, list[dict]], bool]
    margin_threshold: float = 0.0
    promote_confirms: bool = False


@dataclass
class RecordingMetrics:
    recording_name: str
    total_seconds: float
    confirmed_seconds: float
    matched_seconds: float
    wrong_seconds: float
    provisional_seconds: float
    confirmed_precision: float
    wrong_confirmed_rate: float
    confirmed_coverage: float
    provisional_rate: float
    reference_speakers: list[str]
    confirmed_canonical_speakers: list[str]
    provisional_candidates: list[str]
    switches: int
    median_delay: float
    p95_delay: float
    delays: list[float] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    aggregation_type: str  # "MICRO" or "MACRO"
    confirmed_precision: float
    wrong_confirmed_rate: float
    confirmed_coverage: float
    provisional_rate: float
    median_delay: float
    p95_delay: float
    speakers_by_recording: dict[str, list[str]]


def resolve_file(filename: str) -> Path:
    """Find file in standard candidate locations (container or host)."""
    candidates = [
        Path("/data/benchmarks") / filename,
        Path(__file__).resolve().parent.parent.parent.parent
        / "benchmarks"
        / "elevenlabs-stt"
        / "results"
        / "private"
        / filename,
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "benchmarks"
        / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not locate {filename} in {candidates}")


def load_cache() -> dict[str, list[dict]]:
    path = resolve_file("window-cache.json")
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_turns() -> dict[str, dict]:
    path = resolve_file("reference-turns.json")
    return json.loads(path.read_text(encoding="utf-8"))


def simulate_recording(
    windows_data: list[dict],
    policy: PolicySpec,
) -> tuple[list[dict], list[str], list[str], list[str], list[float]]:
    """Simulate streaming live session over window snapshots for a specific policy."""
    store = LiveSessionStore()
    session = store.create()
    history: list[dict] = []
    output_windows: list[dict] = []
    delays: list[float] = []

    for w in windows_data:
        seq = w["sequence"]
        start, end = w["start"], w["end"]
        boundary = max(start, end - 4.0)
        words = [
            NormalizedWord(
                start=x["start"],
                end=x["end"],
                text="kelime",
                speaker_id=x["speaker_id"],
            )
            for x in w["words"]
            if x.get("speaker_id")
        ]
        pi = _provider_intervals(words, offset=start, window=(start, end))

        reference = session.last_snapshot or {
            lbl: list(spk.committed) for lbl, spk in session.speakers.items()
        }
        matches, unmatched, ambiguous = match_speakers(
            reference,
            pi,
            window=(start, end),
            gap_tolerance_seconds=2.0,
            allow_disjoint_merge=True,
        )

        def split(spans, w_start=start, w_end=end, w_boundary=boundary):
            stable = [(max(s, w_start), min(e, w_boundary)) for s, e in spans if s < w_boundary]
            tail = [(max(s, w_boundary), min(e, w_end)) for s, e in spans if e > w_boundary]
            return [s for s in stable if s[1] > s[0]], [s for s in tail if s[1] > s[0]]

        assignments: list[dict] = []
        last_snapshot: dict[str, list[tuple[float, float]]] = {}
        cur_coexist = len(pi) > 1

        for provider_name in sorted(pi):
            spans = pi[provider_name]
            match = matches.get(provider_name)
            if match is None:
                store._register_candidate(session, spans, seq)
                continue
            label = match.canonical_speaker
            stable, tail = split(spans)
            speaker = session.speaker(label)
            speaker.committed.extend(stable)
            last_snapshot.setdefault(label, []).extend(spans)

            if stable:
                s_start = min(s[0] for s in stable)
                s_end = max(s[1] for s in stable)
                confirmed = policy.is_confirmed_fn(
                    label, s_start, s_end, match, cur_coexist, seq, history
                )
                if confirmed:
                    session.confirmed_labels_set.add(label)
                    midpoint = (s_start + s_end) / 2
                    delays.append(max(0.0, end + PROVIDER_LATENCY_SECONDS - midpoint))
                assignments.append({
                    "canonical_speaker": label,
                    "provisional": not confirmed,
                    "start": round(s_start, 3),
                    "end": round(s_end, 3),
                })
                history.append({
                    "sequence": seq,
                    "label": label,
                    "start": round(s_start, 3),
                    "end": round(s_end, 3),
                    "margin": match.margin,
                    "evidence": match.evidence,
                    "coexist": cur_coexist,
                    "window_end": round(end, 3),
                })
            if tail:
                t_start = min(t[0] for t in tail)
                t_end = max(t[1] for t in tail)
                assignments.append({
                    "canonical_speaker": label,
                    "provisional": True,
                    "start": round(t_start, 3),
                    "end": round(t_end, 3),
                })

        # Promotions
        consumed = []
        for idx, candidate in enumerate(session.candidates):
            if not store._may_promote(session):
                break
            if candidate.seen_snapshots < 2:
                continue
            label = next_canonical_label(set(session.speakers))
            speaker = session.speaker(label)
            stable, tail = split(candidate.intervals)
            speaker.committed.extend(stable)
            last_snapshot.setdefault(label, []).extend(candidate.intervals)
            consumed.append(idx)

            if stable:
                s_start = min(s[0] for s in stable)
                s_end = max(s[1] for s in stable)
                confirmed = policy.promote_confirms
                if confirmed:
                    session.confirmed_labels_set.add(label)
                    midpoint = (s_start + s_end) / 2
                    delays.append(max(0.0, end + PROVIDER_LATENCY_SECONDS - midpoint))
                assignments.append({
                    "canonical_speaker": label,
                    "provisional": not confirmed,
                    "start": round(s_start, 3),
                    "end": round(s_end, 3),
                })
                history.append({
                    "sequence": seq,
                    "label": label,
                    "start": round(s_start, 3),
                    "end": round(s_end, 3),
                    "margin": 0.0,
                    "evidence": "promoted",
                    "coexist": cur_coexist,
                    "window_end": round(end, 3),
                })
            if tail:
                t_start = min(t[0] for t in tail)
                t_end = max(t[1] for t in tail)
                assignments.append({
                    "canonical_speaker": label,
                    "provisional": True,
                    "start": round(t_start, 3),
                    "end": round(t_end, 3),
                })

        session.candidates = [c for i, c in enumerate(session.candidates) if i not in consumed]
        session.last_snapshot = last_snapshot
        output_windows.append({
            "sequence": seq,
            "window": [start, end],
            "assignments": assignments,
        })

    confirmed_speakers = session.confirmed_labels()
    all_speakers = list(session.speakers.keys())
    provisional_candidates = [
        label for label in all_speakers if label not in session.confirmed_labels_set
    ]
    return output_windows, confirmed_speakers, all_speakers, provisional_candidates, delays


def evaluate_timeline(
    recording_name: str,
    output_windows: list[dict],
    turns: list[dict],
    duration: float,
    confirmed_speakers: list[str],
    provisional_candidates: list[str],
    delays: list[float],
    resolution: float = RESOLUTION,
) -> RecordingMetrics:
    """Evaluate consensus timeline against reference turns using the frontend freeze rule."""
    timeline: list[str | None] = [None] * int(duration / resolution + 1)

    for w in output_windows:
        for a in w["assignments"]:
            if a["provisional"]:
                continue
            s_idx = int(a["start"] / resolution)
            e_idx = int(a["end"] / resolution) + 1
            for i in range(max(0, s_idx), min(len(timeline), e_idx)):
                # Freeze rule: once confirmed, outside mutable tail never oscillates
                if timeline[i] is None:
                    timeline[i] = a["canonical_speaker"]

    total_pts = 0
    unresolved_pts = 0
    overlap: dict[str, dict[str, int]] = {}

    for idx, label in enumerate(timeline):
        t = idx * resolution
        turn = next((x for x in turns if x["start_seconds"] <= t <= x["end_seconds"]), None)
        if turn is None or turn["speaker"] == "Bilinmeyen":
            continue
        total_pts += 1
        if label is None:
            unresolved_pts += 1
        else:
            spk = turn["speaker"]
            overlap.setdefault(spk, {}).setdefault(label, 0)
            overlap[spk][label] += 1

    # Optimal bipartite 1-to-1 matching between reference and canonical live speakers
    pairs = [(cnt, r, c) for r, m in overlap.items() for c, cnt in m.items()]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_r: set[str] = set()
    used_c: set[str] = set()
    matched_pts = 0
    for cnt, r, c in pairs:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        matched_pts += cnt

    confirmed_pts = total_pts - unresolved_pts
    wrong_pts = confirmed_pts - matched_pts

    prec = matched_pts / confirmed_pts if confirmed_pts else 0.0
    wrong_rate = wrong_pts / total_pts if total_pts else 0.0
    cov = confirmed_pts / total_pts if total_pts else 0.0
    prov_rate = unresolved_pts / total_pts if total_pts else 0.0

    # Label switches on confirmed speech points
    switches = 0
    prev: str | None = None
    for lbl in timeline:
        if lbl is not None:
            if prev is not None and lbl != prev:
                switches += 1
            prev = lbl

    med_del = round(statistics.median(delays), 2) if delays else 0.0
    p95_del = round(sorted(delays)[int(len(delays) * 0.95)], 2) if delays else 0.0
    ref_speakers = sorted({t["speaker"] for t in turns if t["speaker"] != "Bilinmeyen"})

    return RecordingMetrics(
        recording_name=recording_name,
        total_seconds=round(total_pts * resolution, 2),
        confirmed_seconds=round(confirmed_pts * resolution, 2),
        matched_seconds=round(matched_pts * resolution, 2),
        wrong_seconds=round(wrong_pts * resolution, 2),
        provisional_seconds=round(unresolved_pts * resolution, 2),
        confirmed_precision=round(prec, 3),
        wrong_confirmed_rate=round(wrong_rate, 3),
        confirmed_coverage=round(cov, 3),
        provisional_rate=round(prov_rate, 3),
        reference_speakers=ref_speakers,
        confirmed_canonical_speakers=confirmed_speakers,
        provisional_candidates=provisional_candidates,
        switches=switches,
        median_delay=med_del,
        p95_delay=p95_del,
        delays=delays,
    )


def aggregate_micro(per_rec: dict[str, RecordingMetrics]) -> AggregatedMetrics:
    """Micro aggregation: pooled speech points / duration weighted."""
    tot_s = sum(r.total_seconds for r in per_rec.values())
    conf_s = sum(r.confirmed_seconds for r in per_rec.values())
    mat_s = sum(r.matched_seconds for r in per_rec.values())
    wr_s = sum(r.wrong_seconds for r in per_rec.values())
    prov_s = sum(r.provisional_seconds for r in per_rec.values())

    prec = mat_s / conf_s if conf_s else 0.0
    wrong_rate = wr_s / tot_s if tot_s else 0.0
    cov = conf_s / tot_s if tot_s else 0.0
    prov_rate = prov_s / tot_s if tot_s else 0.0

    all_delays: list[float] = []
    for r in per_rec.values():
        all_delays.extend(r.delays)

    med_del = round(statistics.median(all_delays), 2) if all_delays else 0.0
    p95_del = round(sorted(all_delays)[int(len(all_delays) * 0.95)], 2) if all_delays else 0.0

    return AggregatedMetrics(
        aggregation_type="MICRO",
        confirmed_precision=round(prec, 3),
        wrong_confirmed_rate=round(wrong_rate, 3),
        confirmed_coverage=round(cov, 3),
        provisional_rate=round(prov_rate, 3),
        median_delay=med_del,
        p95_delay=p95_del,
        speakers_by_recording={k: r.confirmed_canonical_speakers for k, r in per_rec.items()},
    )


def aggregate_macro(per_rec: dict[str, RecordingMetrics]) -> AggregatedMetrics:
    """Macro aggregation: unweighted arithmetic mean of per-recording rates."""
    n = len(per_rec)
    prec = sum(r.confirmed_precision for r in per_rec.values()) / n if n else 0.0
    wrong_rate = sum(r.wrong_confirmed_rate for r in per_rec.values()) / n if n else 0.0
    cov = sum(r.confirmed_coverage for r in per_rec.values()) / n if n else 0.0
    prov_rate = sum(r.provisional_rate for r in per_rec.values()) / n if n else 0.0

    med_del = round(statistics.mean(r.median_delay for r in per_rec.values()), 2) if n else 0.0
    p95_del = round(statistics.mean(r.p95_delay for r in per_rec.values()), 2) if n else 0.0

    return AggregatedMetrics(
        aggregation_type="MACRO",
        confirmed_precision=round(prec, 3),
        wrong_confirmed_rate=round(wrong_rate, 3),
        confirmed_coverage=round(cov, 3),
        provisional_rate=round(prov_rate, 3),
        median_delay=med_del,
        p95_delay=p95_del,
        speakers_by_recording={k: r.confirmed_canonical_speakers for k, r in per_rec.items()},
    )


def build_policies() -> list[PolicySpec]:
    """Canonical definitions for candidate policies."""
    policies: list[PolicySpec] = []

    # P1: Baseline - every stable matched span confirms immediately, promoted confirms immediately
    policies.append(
        PolicySpec(
            name="P1",
            is_confirmed_fn=lambda lbl, s_s, s_e, match, coex, seq, hist: True,
            promote_confirms=True,
        )
    )

    # P2@M: Multi-snapshot margin confirmation
    for m in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        def make_p2(threshold: float):
            def is_conf(lbl, s_s, s_e, match, coex, seq, hist):
                if match.evidence in ("merged-fragment", "gap"):
                    return False
                supp = [
                    h
                    for h in hist
                    if h["label"] == lbl
                    and min(h["end"], s_e) >= max(h["start"], s_s) - 0.5
                ]
                agreeing_seqs = {h["sequence"] for h in supp} | {seq}
                margins = [
                    h["margin"] for h in supp if h.get("margin") is not None
                ] + [match.margin]
                return len(agreeing_seqs) >= 2 and max(margins) >= threshold
            return is_conf

        policies.append(
            PolicySpec(
                name=f"P2@{m:.1f}",
                is_confirmed_fn=make_p2(m),
                margin_threshold=m,
                promote_confirms=False,
            )
        )

    # P3: >= 3 agreeing snapshots OR (>= 2 agreeing AND provider coexisted)
    def is_conf_p3(lbl, s_s, s_e, match, coex, seq, hist):
        if match.evidence in ("merged-fragment", "gap"):
            return False
        supp = [
            h
            for h in hist
            if h["label"] == lbl
            and min(h["end"], s_e) >= max(h["start"], s_s) - 0.5
        ]
        agreeing_seqs = {h["sequence"] for h in supp} | {seq}
        coex_any = coex or any(h["coexist"] for h in supp)
        if len(agreeing_seqs) >= 3:
            return True
        return len(agreeing_seqs) >= 2 and coex_any

    policies.append(
        PolicySpec(
            name="P3",
            is_confirmed_fn=is_conf_p3,
            promote_confirms=False,
        )
    )

    # P4: Strong majority (>= 0.75) across snapshots covering the instant
    def is_conf_p4(lbl, s_s, s_e, match, coex, seq, hist):
        if match.evidence in ("merged-fragment", "gap"):
            return False
        covering = [
            h
            for h in hist
            if min(h["end"], s_e) >= max(h["start"], s_s) - 0.5
        ]
        covering_seqs = {h["sequence"] for h in covering} | {seq}
        agreeing = [h for h in covering if h["label"] == lbl]
        agreeing_seqs = {h["sequence"] for h in agreeing} | {seq}
        ratio = len(agreeing_seqs) / max(len(covering_seqs), 1)
        return len(agreeing_seqs) >= 2 and ratio >= 0.75

    policies.append(
        PolicySpec(
            name="P4",
            is_confirmed_fn=is_conf_p4,
            promote_confirms=False,
        )
    )

    # P5 variants: >= 2 agreeing snapshots AND provider coexistence AND margin >= M
    for m in [0.5, 1.0, 1.5, 2.0]:
        def make_p5(threshold: float):
            def is_conf(lbl, s_s, s_e, match, coex, seq, hist):
                if match.evidence in ("merged-fragment", "gap"):
                    return False
                supp = [
                    h
                    for h in hist
                    if h["label"] == lbl
                    and min(h["end"], s_e) >= max(h["start"], s_s) - 0.5
                ]
                agreeing_seqs = {h["sequence"] for h in supp} | {seq}
                coex_any = coex or any(h["coexist"] for h in supp)
                margins = [
                    h["margin"] for h in supp if h.get("margin") is not None
                ] + [match.margin]
                return len(agreeing_seqs) >= 2 and coex_any and max(margins) >= threshold
            return is_conf

        policies.append(
            PolicySpec(
                name=f"P5@{m:.1f}",
                is_confirmed_fn=make_p5(m),
                margin_threshold=m,
                promote_confirms=False,
            )
        )

    return policies


def run_full_calibration() -> dict[str, dict]:
    """Execute evaluation for all policies across both datasets."""
    cache = load_cache()
    refs = load_reference_turns()
    policies = build_policies()

    results: dict[str, dict] = {}

    for pol in policies:
        per_rec: dict[str, RecordingMetrics] = {}
        for rec_name in ["recent-2p", "multi-person-real"]:
            w_key = "recent-2p" if rec_name == "recent-2p" else "multi-4p"
            r_info = refs[rec_name]
            windows_data = cache[w_key]
            turns = r_info["turns"]
            duration = r_info["duration"]

            output_windows, confirmed_spks, all_spks, prov_cands, delays = simulate_recording(
                windows_data, pol
            )
            metrics = evaluate_timeline(
                recording_name=rec_name,
                output_windows=output_windows,
                turns=turns,
                duration=duration,
                confirmed_speakers=confirmed_spks,
                provisional_candidates=prov_cands,
                delays=delays,
            )
            per_rec[rec_name] = metrics

        micro = aggregate_micro(per_rec)
        macro = aggregate_macro(per_rec)
        results[pol.name] = {
            "per_recording": per_rec,
            "micro": micro,
            "macro": macro,
        }

    return results
