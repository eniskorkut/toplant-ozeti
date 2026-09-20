#!/usr/bin/env python3
"""Conservative live diarization calibration study.

Uses canonical_evaluator as the single source of truth to evaluate policies
offline on cached windows against reference transcripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from canonical_evaluator import run_full_calibration


def main() -> None:
    results = run_full_calibration()

    print("=" * 110)
    print("PER-RECORDING METRICS (recent-2p vs multi-person-real)")
    print("=" * 110)
    header = (
        f"{'Policy':<8} | {'Recording':<17} | {'Ref Spks':<10} | {'Live Conf':<20} | "
        f"{'Live Prov':<20} | {'Prec':>6} | {'WrRate':>6} | {'Cov':>6} | {'ProvRate':>8} | "
        f"{'Sw':>3} | {'MedDel':>6} | {'P95Del':>6}"
    )
    print(header)
    print("-" * len(header))
    for name, data in results.items():
        for rec_name in ["recent-2p", "multi-person-real"]:
            r = data["per_recording"][rec_name]
            print(
                f"{name:<8} | {rec_name:<17} | {str(r.reference_speakers):<10} | "
                f"{str(r.confirmed_canonical_speakers):<20} | {str(r.provisional_candidates):<20} | "
                f"{r.confirmed_precision:>6.1%} | {r.wrong_confirmed_rate:>6.1%} | "
                f"{r.confirmed_coverage:>6.1%} | {r.provisional_rate:>8.1%} | "
                f"{r.switches:>3} | {r.median_delay:>5.2f}s | {r.p95_delay:>5.2f}s"
            )

    print("\n" + "=" * 110)
    print("MICRO AGGREGATED METRICS (Duration-Weighted Pooling)")
    print("=" * 110)
    mic_header = (
        f"{'Policy':<8} | {'Prec (Micro)':>12} | {'WrRate':>8} | {'Coverage':>8} | "
        f"{'Provisional':>11} | {'Med Delay':>9} | {'P95 Delay':>9} | {'Confirmed Speakers (2p / multi)':<35}"
    )
    print(mic_header)
    print("-" * len(mic_header))
    for name, data in results.items():
        m = data["micro"]
        p2 = data["per_recording"]["recent-2p"]
        p4 = data["per_recording"]["multi-person-real"]
        spks = f"{len(p2.confirmed_canonical_speakers)} in 2p / {len(p4.confirmed_canonical_speakers)} in multi"
        print(
            f"{name:<8} | {m.confirmed_precision:>12.1%} | {m.wrong_confirmed_rate:>8.1%} | "
            f"{m.confirmed_coverage:>8.1%} | {m.provisional_rate:>11.1%} | "
            f"{m.median_delay:>8.2f}s | {m.p95_delay:>8.2f}s | {spks:<35}"
        )

    print("\n" + "=" * 110)
    print("MACRO AGGREGATED METRICS (Unweighted Dataset Average)")
    print("=" * 110)
    mac_header = (
        f"{'Policy':<8} | {'Prec (Macro)':>12} | {'WrRate':>8} | {'Coverage':>8} | "
        f"{'Provisional':>11} | {'Med Delay':>9} | {'P95 Delay':>9} | {'Confirmed Speakers (2p / multi)':<35}"
    )
    print(mac_header)
    print("-" * len(mac_header))
    for name, data in results.items():
        m = data["macro"]
        p2 = data["per_recording"]["recent-2p"]
        p4 = data["per_recording"]["multi-person-real"]
        spks = f"{len(p2.confirmed_canonical_speakers)} in 2p / {len(p4.confirmed_canonical_speakers)} in multi"
        print(
            f"{name:<8} | {m.confirmed_precision:>12.1%} | {m.wrong_confirmed_rate:>8.1%} | "
            f"{m.confirmed_coverage:>8.1%} | {m.provisional_rate:>11.1%} | "
            f"{m.median_delay:>8.2f}s | {m.p95_delay:>8.2f}s | {spks:<35}"
        )


if __name__ == "__main__":
    main()
