#!/usr/bin/env python3
"""Conservative-confirmation policy study for live speaker attribution.

Delegates to canonical_evaluator as the single source of truth across the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from canonical_evaluator import (
    PRIVATE_DIR,
    run_full_calibration,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-thresholds", default="0.5,1.0,2.0")
    parser.add_argument("--reuse", action="store_true", help="evaluate stored history only")
    args = parser.parse_args()

    results = run_full_calibration()
    report = {
        "policies": {
            name: {
                "combined": {
                    "precision": data["micro"].confirmed_precision,
                    "wrong_confirmed_rate": data["micro"].wrong_confirmed_rate,
                    "confirmed_coverage": data["micro"].confirmed_coverage,
                    "provisional_rate": data["micro"].provisional_rate,
                    "median_confirmation_delay_seconds": data["micro"].median_delay,
                },
                "macro": {
                    "precision": data["macro"].confirmed_precision,
                    "wrong_confirmed_rate": data["macro"].wrong_confirmed_rate,
                    "confirmed_coverage": data["macro"].confirmed_coverage,
                    "provisional_rate": data["macro"].provisional_rate,
                },
                "per_recording": {
                    r_name: {
                        "confirmed_precision": r.confirmed_precision,
                        "wrong_confirmed_rate": r.wrong_confirmed_rate,
                        "confirmed_coverage": r.confirmed_coverage,
                        "provisional_rate": r.provisional_rate,
                        "confirmed_canonical_speakers": r.confirmed_canonical_speakers,
                        "provisional_candidates": r.provisional_candidates,
                        "switches": r.switches,
                        "median_delay": r.median_delay,
                    }
                    for r_name, r in data["per_recording"].items()
                },
            }
            for name, data in results.items()
        }
    }

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = PRIVATE_DIR / "conservative-calibration.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration metrics saved to {out_file}", flush=True)

    for name, p_data in report["policies"].items():
        mic = p_data["combined"]
        print(
            f"{name:<8} | Micro Prec={mic['precision']:>5.1%} Wr={mic['wrong_confirmed_rate']:>5.1%} "
            f"Cov={mic['confirmed_coverage']:>5.1%} Prov={mic['provisional_rate']:>5.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
