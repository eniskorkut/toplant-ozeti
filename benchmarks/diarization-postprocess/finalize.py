#!/usr/bin/env python3
"""Combine sweep metrics + dscore scoring, apply the selection rules and emit reports.

Host-side, standard library only. Safe `aggregate.json` (no transcript text) is written
next to the harness; the detailed markdown goes to the git-ignored results directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selection import combined_assignment_error, rank_key, select_candidate

HARNESS = Path(__file__).resolve().parent
RESULTS = HARNESS / "results"
BASELINE_KEY = "on0.30_off0.50"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_metrics_and_scoring(metrics: dict, scoring: dict) -> list[dict]:
    entries = []
    for key, aggregate in metrics["configs"].items():
        scored = scoring["configs"].get(key, {})
        entries.append(
            {
                **aggregate,
                "der": scored.get("der"),
                "jer": scored.get("jer"),
                "der_overlap_ignored": scored.get("der_overlap_ignored"),
            }
        )
    return entries


def table_row(entry: dict) -> str:
    return (
        f"| {entry['config']} | {entry.get('der', float('nan')):.2f}% | {entry.get('jer', float('nan')):.2f}% | "
        f"{entry.get('der_overlap_ignored', float('nan')):.2f}% | {entry.get('speaker_count_accuracy', 0):.2%} | "
        f"{entry.get('speaker_count_mae', float('nan')):.3f} | {entry['coverage_ratio']:.1%} | "
        f"{entry['overlap_ratio']:.1%} | {entry['unresolved_words']} | {entry['unresolved_rate']:.2%} | "
        f"{entry['wrong_attribution_words']} | {entry['wrong_attribution_rate']:.2%} | "
        f"{combined_assignment_error(entry):.2%} | {entry['speaker_turns']} | {entry['rapid_flips']} |"
    )


def header() -> list[str]:
    return [
        "| Config | DER | JER | DER(no overlap) | Count acc | Count MAE | Coverage | Overlap | Unresolved | Unresolved rate | Wrong | Wrong rate | Combined error | Turns | Flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibration", "validation"], default="calibration")
    args = parser.parse_args()

    metrics = load(RESULTS / f"metrics-{args.stage}.json")
    scoring = load(RESULTS / f"scoring-{args.stage}.json")
    entries = merge_metrics_and_scoring(metrics, scoring)
    baseline = next((entry for entry in entries if entry["config"] == BASELINE_KEY), None)
    if baseline is None:
        raise SystemExit("baseline config on0.30_off0.50 missing from the sweep")

    lines = [
        f"# Diarization post-processing sweep ({args.stage})",
        "",
        f"Fixed: TitaNet Small, threshold {metrics['fixed']['threshold']}, "
        f"{metrics['fixed']['threads']} threads, automatic speaker count, "
        f"production merge tolerance {metrics['fixed']['merge_tolerance']} s, "
        "current heuristic STT timestamps.",
        "",
        *header(),
    ]
    for entry in sorted(entries, key=lambda item: rank_key(item, current_on=0.3, current_off=0.5)):
        lines.append(table_row(entry))

    output: dict = {
        "stage": args.stage,
        "fixed": metrics["fixed"],
        "files": metrics["files"],
        "entries": entries,
        "baseline": baseline,
    }

    if args.stage == "calibration":
        result = select_candidate(baseline, entries)
        output["selection"] = {
            "baseline_config": BASELINE_KEY,
            "eligible": result["eligible"],
            "selected": result["selected"]["config"] if result["selected"] else None,
            "selected_parameters": {
                "min_duration_on": result["selected"]["min_duration_on"],
                "min_duration_off": result["selected"]["min_duration_off"],
            }
            if result["selected"]
            else None,
            "rank_key": list(rank_key(result["selected"], current_on=0.3, current_off=0.5))
            if result["selected"]
            else None,
            "reason": result["reason"],
        }
        lines += [
            "",
            "## Selection (calibration only)",
            "",
            f"- baseline: `{BASELINE_KEY}` (wrong-attribution tolerance +0.005 absolute)",
            f"- eligible: {', '.join(result['eligible']) or 'none'}",
            f"- **selected: {output['selection']['selected']}** ({result['reason']})",
        ]

    (RESULTS / f"report-{args.stage}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HARNESS / f"aggregate-{args.stage}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote aggregate-{args.stage}.json and results/report-{args.stage}.md")
    if args.stage == "calibration":
        print("selected:", output["selection"]["selected"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
