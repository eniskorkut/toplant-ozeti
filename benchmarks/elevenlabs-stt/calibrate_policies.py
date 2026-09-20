#!/usr/bin/env python3
"""Conservative live diarization calibration study.

Evaluates candidate confirmation policies offline on cached 12s/4s windows from
recent-2p and multi-4p recordings against their final authoritative full-file
Scribe v2 transcripts.

Reports:
- confirmed precision (correct confirmed / all confirmed) [PRIMARY METRIC]
- wrong confirmed attribution rate
- confirmed coverage
- provisional rate
- median and p95 confirmation delay
- canonical speaker count and label switches
"""

from __future__ import annotations

import json
import statistics
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CACHE_FILE = SCRIPT_DIR / "results" / "private" / "window-cache.json"

RESOLUTION = 0.25
PROVIDER_LATENCY_SECONDS = 1.0  # measured median

RECORDINGS = {
    "recent-2p": {
        "meeting_id": "0eb33b95c5874947ba19f1e196fe2fec",
        "final_meeting_id": "0eb33b95c5874947ba19f1e196fe2fec",
    },
    "multi-4p": {
        "meeting_id": "b1095740120b4b1e96db337da961ea63",
        "final_meeting_id": "diar-final-v2",
    },
}


def words_to_pi(words: list[dict], offset: float, window: tuple[float, float]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = {}
    for w in words:
        spk = w.get("speaker_id")
        if not spk:
            continue
        start = max(window[0], offset + w["start"])
        end = min(window[1], offset + w["end"])
        if end <= start:
            continue
        intervals.setdefault(spk, []).append((start, end))
    return intervals


def get_final_turns(meeting_id: str) -> list[dict]:
    url = f"http://localhost:8000/api/v1/meetings/{meeting_id}/transcript"
    req = urllib.request.urlopen(url)
    return json.loads(req.read().decode("utf-8"))["turns"]


def run_session_with_policy(windows_data: list[dict], policy_name: str, margin_threshold: float = 0.0) -> dict:
    """Simulate live session processing for a specific policy."""
    from app.services.live_sessions import LiveSessionStore, LiveSession, next_canonical_label
    from app.services.speaker_matching import match_speakers, total_seconds

    store = LiveSessionStore()
    session = store.create()

    # Track evidence history per speech segment:
    # list of {sequence, label, start, end, margin, evidence, coexist, window_end}
    history: list[dict] = []
    output_windows: list[dict] = []
    label_delays: list[float] = []

    def is_confirmed(label: str, s_start: float, s_end: float, cur_margin: float, cur_coexist: bool, cur_seq: int, evidence: str) -> bool:
        if policy_name == "P1":
            # Current baseline: any stable matched span confirms immediately
            return True

        # Never confirm weak merged fragments
        if evidence == "merged-fragment":
            return False

        # Find supporting snapshot sequences overlapping [s_start, s_end]
        supporting = [
            h for h in history
            if h["label"] == label
            and min(h["end"], s_end) >= max(h["start"], s_start) - 0.5
        ]
        agreeing_seqs = {h["sequence"] for h in supporting}
        # Cur sequence is also agreeing
        agreeing_seqs.add(cur_seq)
        
        coexist_any = cur_coexist or any(h["coexist"] for h in supporting)
        all_margins = [h["margin"] for h in supporting if h.get("margin") is not None]
        if cur_margin is not None:
            all_margins.append(cur_margin)
        max_margin = max(all_margins) if all_margins else 0.0

        if policy_name.startswith("P2"):
            # Require >= 2 agreeing snapshots AND margin >= threshold
            # Note: promoted speakers were verified across >= 2 snapshots, exempt from overlap margin
            margin_ok = max_margin >= margin_threshold or evidence == "promoted"
            return len(agreeing_seqs) >= 2 and margin_ok

        if policy_name == "P3":
            # Require >= 3 agreeing snapshots OR (>= 2 agreeing AND provider coexisted)
            if len(agreeing_seqs) >= 3:
                return True
            if len(agreeing_seqs) >= 2 and coexist_any:
                return True
            return False

        if policy_name == "P4":
            # Find all snapshots covering [s_start, s_end]
            covering = [
                h for h in history
                if min(h["end"], s_end) >= max(h["start"], s_start) - 0.5
            ]
            covering_seqs = {h["sequence"] for h in covering}
            covering_seqs.add(cur_seq)
            agreement_ratio = len(agreeing_seqs) / max(len(covering_seqs), 1)
            return len(agreeing_seqs) >= 2 and agreement_ratio >= 0.75

        if policy_name.startswith("P5"):
            # Require >= 2 agreeing snapshots AND coexist in at least one snapshot AND margin
            margin_ok = max_margin >= margin_threshold or evidence == "promoted"
            return len(agreeing_seqs) >= 2 and coexist_any and margin_ok

        return False

    for w in windows_data:
        seq = w["sequence"]
        start, end = w["start"], w["end"]
        boundary = max(start, end - 4.0)
        pi = words_to_pi(w["words"], start, (start, end))

        # Build reference
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

        assignments = []
        last_snapshot: dict[str, list[tuple[float, float]]] = {}

        def split(spans):
            stable = [(max(s, start), min(e, boundary)) for s, e in spans if s < boundary]
            tail = [(max(s, boundary), min(e, end)) for s, e in spans if e > boundary]
            return [s for s in stable if s[1] > s[0]], [s for s in tail if s[1] > s[0]]

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

            cur_coexist = len(pi) > 1
            if stable:
                s_start = min(s[0] for s in stable)
                s_end = max(s[1] for s in stable)
                confirmed = is_confirmed(label, s_start, s_end, match.margin, cur_coexist, seq, match.evidence)
                if confirmed:
                    session.confirmed_labels_set.add(label)
                    midpoint = (s_start + s_end) / 2
                    label_delays.append(max(0.0, end + PROVIDER_LATENCY_SECONDS - midpoint))
                
                assignments.append({
                    "label": label,
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
                    "label": label,
                    "provisional": True,
                    "start": round(t_start, 3),
                    "end": round(t_end, 3),
                })

        # Promotions
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

            cur_coexist = len(pi) > 1
            if stable:
                s_start = min(s[0] for s in stable)
                s_end = max(s[1] for s in stable)
                confirmed = is_confirmed(label, s_start, s_end, 0.0, cur_coexist, seq, "promoted")
                if confirmed:
                    session.confirmed_labels_set.add(label)
                    midpoint = (s_start + s_end) / 2
                    label_delays.append(max(0.0, end + PROVIDER_LATENCY_SECONDS - midpoint))

                assignments.append({
                    "label": label,
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
                    "label": label,
                    "provisional": True,
                    "start": round(t_start, 3),
                    "end": round(t_end, 3),
                })

        session.candidates = [c for i, c in enumerate(session.candidates) if c.seen_snapshots < 2]
        session.last_snapshot = last_snapshot

        output_windows.append({
            "sequence": seq,
            "window": [start, end],
            "assignments": assignments,
        })

    return {
        "windows": output_windows,
        "confirmed_speakers": session.confirmed_labels(),
        "total_speakers": list(session.speakers.keys()),
        "label_delays": label_delays,
    }


def consensus_seconds(windows: list[dict], duration: float, resolution: float = RESOLUTION) -> list[str | None]:
    labels = [None] * int(duration / resolution + 1)
    for w in windows:
        for a in w["assignments"]:
            if a["provisional"]:
                continue
            start_index = int(a["start"] / resolution)
            end_index = int(a["end"] / resolution) + 1
            for idx in range(max(0, start_index), min(len(labels), end_index)):
                labels[idx] = a["label"]
    return labels


def compare_to_final(windows: list[dict], turns: list[dict], duration: float, resolution: float = RESOLUTION) -> dict:
    labels = consensus_seconds(windows, duration, resolution)
    unresolved = 0
    total = 0
    attributed: dict[str, dict[str, int]] = {}

    for idx, label in enumerate(labels):
        time_at = idx * resolution
        turn = next((t for t in turns if t["start_seconds"] <= time_at <= t["end_seconds"]), None)
        if turn is None or turn["speaker"] == "Bilinmeyen":
            continue
        spk = turn["speaker"]
        total += 1
        if label is None:
            unresolved += 1
        else:
            attributed.setdefault(spk, {}).setdefault(label, 0)
            attributed[spk][label] += 1

    pairs = []
    for final_spk, counts in attributed.items():
        for canonical, count in counts.items():
            pairs.append((count, final_spk, canonical))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_final: set[str] = set()
    used_canonical: set[str] = set()
    matched = 0
    for count, final_spk, canonical in pairs:
        if final_spk in used_final or canonical in used_canonical:
            continue
        used_final.add(final_spk)
        used_canonical.add(canonical)
        matched += count

    confirmed = total - unresolved
    wrong = total - matched - unresolved
    precision = matched / confirmed if confirmed else 0.0

    # Count label switches on timeline
    switches = 0
    prev: str | None = None
    for lbl in labels:
        if lbl is not None:
            if prev is not None and lbl != prev:
                switches += 1
            prev = lbl

    return {
        "total": total,
        "matched": matched,
        "wrong": wrong,
        "unresolved": unresolved,
        "confirmed": confirmed,
        "precision": round(precision, 3),
        "wrong_rate": round(wrong / total, 3) if total else 0.0,
        "confirmed_coverage": round(confirmed / total, 3) if total else 0.0,
        "provisional_rate": round(unresolved / total, 3) if total else 0.0,
        "switches": switches,
    }


def main():
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    turns_by_rec = {name: get_final_turns(spec["final_meeting_id"]) for name, spec in RECORDINGS.items()}
    durations = {name: max(t["end_seconds"] for t in turns) for name, turns in turns_by_rec.items()}

    policies = [
        ("P1", 0.0),
        ("P2@0.5", 0.5),
        ("P2@1.0", 1.0),
        ("P2@1.5", 1.5),
        ("P2@2.0", 2.0),
        ("P2@2.5", 2.5),
        ("P2@3.0", 3.0),
        ("P3", 0.0),
        ("P4", 0.0),
        ("P5@1.0", 1.0),
        ("P5@1.5", 1.5),
        ("P5@2.0", 2.0),
    ]

    results = {}

    print("\n" + "=" * 80)
    print("CALIBRATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Policy':<10} | {'Precision':>9} | {'Wrong':>7} | {'Coverage':>9} | {'Provisional':>11} | {'Med Delay':>9} | {'P95 Delay':>9} | {'Speakers':>8}")
    print("-" * 88)

    for pol_name, threshold in policies:
        combined_total = 0
        combined_matched = 0
        combined_wrong = 0
        combined_unresolved = 0
        all_delays = []
        per_rec = {}

        for name, spec in RECORDINGS.items():
            sim = run_session_with_policy(cache[name], pol_name, threshold)
            comp = compare_to_final(sim["windows"], turns_by_rec[name], durations[name])
            comp["confirmed_speakers"] = sim["confirmed_speakers"]
            comp["total_speakers"] = sim["total_speakers"]
            comp["label_delays"] = sim["label_delays"]
            all_delays.extend(sim["label_delays"])
            per_rec[name] = comp

            combined_total += comp["total"]
            combined_matched += comp["matched"]
            combined_wrong += comp["wrong"]
            combined_unresolved += comp["unresolved"]

        confirmed_tot = combined_total - combined_unresolved
        prec = combined_matched / confirmed_tot if confirmed_tot else 0.0
        wrong_rt = combined_wrong / combined_total if combined_total else 0.0
        cov_rt = confirmed_tot / combined_total if combined_total else 0.0
        prov_rt = combined_unresolved / combined_total if combined_total else 0.0
        med_del = round(statistics.median(all_delays), 2) if all_delays else 0.0
        p95_del = round(sorted(all_delays)[int(len(all_delays) * 0.95)], 2) if all_delays else 0.0

        results[pol_name] = {
            "precision": round(prec, 3),
            "wrong": round(wrong_rt, 3),
            "confirmed_coverage": round(cov_rt, 3),
            "provisional": round(prov_rt, 3),
            "median_delay": med_del,
            "p95_delay": p95_del,
            "per_recording": per_rec,
        }

        print(f"{pol_name:<10} | {prec:>9.1%} | {wrong_rt:>7.1%} | {cov_rt:>9.1%} | {prov_rt:>11.1%} | {med_del:>8.2f}s | {p95_del:>8.2f}s | {len(per_rec['recent-2p']['confirmed_speakers'])}+{len(per_rec['multi-4p']['confirmed_speakers'])}")

    return results


if __name__ == "__main__":
    main()
