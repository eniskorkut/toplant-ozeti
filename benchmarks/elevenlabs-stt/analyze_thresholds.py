#!/usr/bin/env python3
"""Structural analysis of the threshold sweep: cluster sizes, tiny-cluster flags,
candidate ranges for the least-represented cluster, and a human-review suggestion."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
RESULTS = HARNESS / "results"
RAW = RESULTS / "raw"
PRIVATE = RESULTS / "private"

sys.path.insert(0, str(HARNESS))
from el_parse import parse_words  # noqa: E402

THRESHOLDS = [0.22, 0.18, 0.14, 0.10]
TINY_WORD_SHARE = 0.05
TINY_DURATION_SHARE = 0.05


def analyze(threshold: float) -> dict:
    payload = json.loads((RAW / f"threshold-{threshold:.2f}.json").read_text(encoding="utf-8"))
    words = parse_words(payload)

    per_speaker: dict[str, dict] = {}
    turns: list[list] = []
    for word in words:
        speaker = word.speaker or "unassigned"
        entry = per_speaker.setdefault(speaker, {"words": 0, "seconds": 0.0, "turns": [], "ranges": []})
        entry["words"] += 1
        entry["seconds"] += max(0.0, word.end - word.start)
        if turns and turns[-1][0] == speaker:
            turns[-1][2] = word.end
            turns[-1][3] += 1
        else:
            turns.append([speaker, word.start, word.end, 1])

    for speaker, start, end, count in turns:
        per_speaker[speaker]["turns"].append(round(end - start, 3))
        if len(per_speaker[speaker]["ranges"]) < 20:
            per_speaker[speaker]["ranges"].append([round(start, 3), round(end, 3), count])

    total_words = len(words) or 1
    total_seconds = sum(entry["seconds"] for entry in per_speaker.values()) or 1.0
    clusters = {}
    for speaker, entry in sorted(per_speaker.items()):
        median_turn = statistics.median(entry["turns"]) if entry["turns"] else 0.0
        word_share = entry["words"] / total_words
        duration_share = entry["seconds"] / total_seconds
        clusters[speaker] = {
            "words": entry["words"],
            "word_share": round(word_share, 4),
            "seconds": round(entry["seconds"], 2),
            "duration_share": round(duration_share, 4),
            "turns": len(entry["turns"]),
            "median_turn_seconds": round(median_turn, 3),
            "longest_turn_seconds": round(max(entry["turns"], default=0.0), 3),
        }
        flag = (
            "tiny_candidate_cluster"
            if word_share < TINY_WORD_SHARE
            or duration_share < TINY_DURATION_SHARE
            or len(entry["turns"]) <= 1
            else "substantial_cluster"
        )
        clusters[speaker]["flag"] = flag
        clusters[speaker]["ranges"] = entry["ranges"]

    labeled = {k: v for k, v in clusters.items() if k != "unassigned"}
    smallest = min(labeled.items(), key=lambda item: item[1]["words"]) if labeled else None
    return {
        "threshold": threshold,
        "words": len(words),
        "clusters": {k: {kk: vv for kk, vv in v.items() if kk != "ranges"} for k, v in clusters.items()},
        "smallest_cluster": smallest[0] if smallest else None,
        "smallest_cluster_stats": smallest[1] if smallest else None,
        "ranges": {k: v["ranges"] for k, v in clusters.items()},
    }


def candidate_ranges(ranges: list[list], limit: int = 5) -> list[dict]:
    """Prefer ranges with real speech (>= 2 words or >= 0.5 s)."""
    preferred = [r for r in ranges if r[2] >= 2 or (r[1] - r[0]) >= 0.5]
    ordered = sorted(preferred, key=lambda r: r[2], reverse=True)[:limit]
    return [
        {"start": r[0], "end": r[1], "words": r[2], "duration": round(r[1] - r[0], 2)}
        for r in sorted(ordered, key=lambda r: r[0])
    ]


def main() -> int:
    analysis = {f"{t:.2f}": analyze(t) for t in THRESHOLDS}

    PRIVATE.mkdir(parents=True, exist_ok=True)
    candidate_output = {}
    for key, data in analysis.items():
        smallest = data["smallest_cluster"]
        if smallest is None:
            continue
        candidate_output[key] = {
            "cluster": smallest,
            "stats": data["smallest_cluster_stats"],
            "label": "candidate_fourth_cluster" if len(data["clusters"]) >= 4 else "least_represented_cluster",
            "ranges": candidate_ranges(data["ranges"][smallest]),
        }
    (PRIVATE / "fourth-speaker-candidates.json").write_text(
        json.dumps(candidate_output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS / "threshold-analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Safe console summary (counts only, no transcript text).
    for key, data in analysis.items():
        print(f"threshold {key}: speakers={len(data['clusters'])} words={data['words']}")
        for speaker, stats in data["clusters"].items():
            print(
                f"    {speaker}: words={stats['words']} ({stats['word_share']:.1%}) "
                f"seconds={stats['seconds']} ({stats['duration_share']:.1%}) "
                f"turns={stats['turns']} median_turn={stats['median_turn_seconds']}s flag={stats['flag']}"
            )
        candidate = candidate_output.get(key)
        if candidate:
            print(
                f"    least-represented: {candidate['cluster']} "
                f"({candidate['label']}, {len(candidate['ranges'])} ranges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
