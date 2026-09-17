"""Score diarization RTTMs with the pinned dscore checkout (runs in the dscore image).

Usage:
    python /scoring/score_rttm.py --reference ref.rttm [ref2.rttm ...] \
        --system sys.rttm [sys2.rttm ...] [--collar 0.25] [--ignore-overlaps]

Emits one JSON object on stdout. `der`/`jer` are percentages, as dscore defines them.
Per-file missed speech / false alarm / speaker error are not exposed by this scoring
tool version and are therefore reported as unavailable, never invented.
"""

from __future__ import annotations

import argparse
import json

from scorelib.rttm import load_rttm
from scorelib.score import score
from scorelib.turn import merge_turns, trim_turns
from scorelib.uem import gen_uem


def load_many(paths: list[str]) -> list:
    turns = []
    for path in paths:
        file_turns, _, _ = load_rttm(path)
        turns.extend(file_turns)
    return turns


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--system", nargs="+", required=True)
    parser.add_argument("--collar", type=float, default=0.25)
    parser.add_argument("--ignore-overlaps", action="store_true")
    parser.add_argument("--step", type=float, default=0.010)
    parser.add_argument("--jer-min-ref-dur", type=float, default=0.0)
    args = parser.parse_args()

    ref_turns = load_many(args.reference)
    sys_turns = load_many(args.system)
    uem = gen_uem(ref_turns, sys_turns)
    ref_turns = merge_turns(trim_turns(ref_turns, uem))
    sys_turns = merge_turns(trim_turns(sys_turns, uem))

    file_scores, global_scores = score(
        ref_turns,
        sys_turns,
        uem,
        step=args.step,
        collar=args.collar,
        ignore_overlaps=args.ignore_overlaps,
        jer_min_ref_dur=args.jer_min_ref_dur,
    )

    print(
        json.dumps(
            {
                "collar": args.collar,
                "ignore_overlaps": args.ignore_overlaps,
                "global": {"der": global_scores.der, "jer": global_scores.jer},
                "per_file": [
                    {"file_id": entry.file_id, "der": entry.der, "jer": entry.jer}
                    for entry in file_scores
                ],
                "unavailable_metrics": [
                    "missed_speech",
                    "false_alarm",
                    "speaker_error",
                ],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
