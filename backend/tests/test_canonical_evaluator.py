"""Tests for the Canonical Evaluator.

Verifies:
- exact precision calculation
- exact wrong-rate calculation
- provisional calculation
- per-recording speaker count
- micro aggregation
- macro aggregation
- deterministic output
"""

from __future__ import annotations

from app.services.canonical_evaluator import (
    RecordingMetrics,
    aggregate_macro,
    aggregate_micro,
    evaluate_timeline,
    run_full_calibration,
)


def test_exact_precision_wrong_rate_and_provisional_calculation() -> None:
    """Evaluates exact mathematical properties of precision, wrong rate, and provisional rate."""
    turns = [
        {"speaker": "Kişi 1", "start_seconds": 0.0, "end_seconds": 10.0},
        {"speaker": "Kişi 2", "start_seconds": 10.0, "end_seconds": 20.0},
    ]
    output_windows = [
        {
            "sequence": 1,
            "window": [0.0, 10.0],
            "assignments": [
                {
                    "canonical_speaker": "Kişi 1",
                    "provisional": False,
                    "start": 0.0,
                    "end": 8.0,
                },
                {
                    "canonical_speaker": "Kişi 1",
                    "provisional": True,
                    "start": 8.0,
                    "end": 10.0,
                },
            ],
        },
        {
            "sequence": 2,
            "window": [10.0, 20.0],
            "assignments": [
                {
                    "canonical_speaker": "Kişi 1",
                    "provisional": False,
                    "start": 10.0,
                    "end": 15.0,
                },
                {
                    "canonical_speaker": "Kişi 2",
                    "provisional": False,
                    "start": 15.0,
                    "end": 20.0,
                },
            ],
        },
    ]

    metrics = evaluate_timeline(
        recording_name="synthetic-test",
        output_windows=output_windows,
        turns=turns,
        duration=20.0,
        confirmed_speakers=["Kişi 1", "Kişi 2"],
        provisional_candidates=[],
        delays=[5.0, 6.0],
        resolution=0.25,
    )

    # Invariants:
    # confirmed + provisional == total
    assert round(metrics.confirmed_seconds + metrics.provisional_seconds, 2) == round(
        metrics.total_seconds, 2
    )
    # matched + wrong == confirmed
    assert round(metrics.matched_seconds + metrics.wrong_seconds, 2) == round(
        metrics.confirmed_seconds, 2
    )
    # coverage + provisional_rate == 1.0
    assert round(metrics.confirmed_coverage + metrics.provisional_rate, 2) == 1.00
    # precision = matched / confirmed
    expected_precision = round(metrics.matched_seconds / metrics.confirmed_seconds, 3)
    assert metrics.confirmed_precision == expected_precision
    # wrong rate = wrong / total
    expected_wrong_rate = round(metrics.wrong_seconds / metrics.total_seconds, 3)
    assert metrics.wrong_confirmed_rate == expected_wrong_rate


def test_zero_confirmed_points_safe() -> None:
    """When all speech points are provisional, precision is 0.0 and coverage is 0.0."""
    turns = [{"speaker": "Kişi 1", "start_seconds": 0.0, "end_seconds": 5.0}]
    output_windows = [
        {
            "sequence": 1,
            "window": [0.0, 5.0],
            "assignments": [
                {"canonical_speaker": "Kişi 1", "provisional": True, "start": 0.0, "end": 5.0}
            ],
        }
    ]
    metrics = evaluate_timeline(
        recording_name="all-provisional",
        output_windows=output_windows,
        turns=turns,
        duration=5.0,
        confirmed_speakers=[],
        provisional_candidates=["Kişi 1"],
        delays=[],
    )
    assert metrics.confirmed_precision == 0.0
    assert metrics.confirmed_coverage == 0.0
    assert metrics.provisional_rate == 1.0
    assert metrics.wrong_confirmed_rate == 0.0


def test_micro_and_macro_aggregation_exact_math() -> None:
    """Micro weights by speech duration; Macro is unweighted arithmetic mean."""
    rec_a = RecordingMetrics(
        recording_name="rec-A",
        total_seconds=100.0,
        confirmed_seconds=60.0,
        matched_seconds=54.0,
        wrong_seconds=6.0,
        provisional_seconds=40.0,
        confirmed_precision=0.900,  # 54 / 60
        wrong_confirmed_rate=0.060,  # 6 / 100
        confirmed_coverage=0.600,  # 60 / 100
        provisional_rate=0.400,  # 40 / 100
        reference_speakers=["Kişi 1"],
        confirmed_canonical_speakers=["Kişi 1"],
        provisional_candidates=[],
        switches=0,
        median_delay=5.0,
        p95_delay=8.0,
        delays=[5.0] * 60,
    )

    rec_b = RecordingMetrics(
        recording_name="rec-B",
        total_seconds=50.0,
        confirmed_seconds=30.0,
        matched_seconds=21.0,
        wrong_seconds=9.0,
        provisional_seconds=20.0,
        confirmed_precision=0.700,  # 21 / 30
        wrong_confirmed_rate=0.180,  # 9 / 50
        confirmed_coverage=0.600,  # 30 / 50
        provisional_rate=0.400,  # 20 / 50
        reference_speakers=["Kişi 1", "Kişi 2"],
        confirmed_canonical_speakers=["Kişi 1", "Kişi 2"],
        provisional_candidates=[],
        switches=2,
        median_delay=7.0,
        p95_delay=10.0,
        delays=[7.0] * 30,
    )

    per_rec = {"rec-A": rec_a, "rec-B": rec_b}

    micro = aggregate_micro(per_rec)
    assert micro.confirmed_precision == 0.833
    assert micro.wrong_confirmed_rate == 0.100
    assert micro.confirmed_coverage == 0.600
    assert micro.provisional_rate == 0.400

    macro = aggregate_macro(per_rec)
    assert macro.confirmed_precision == 0.800
    assert macro.wrong_confirmed_rate == 0.120
    assert macro.confirmed_coverage == 0.600
    assert macro.provisional_rate == 0.400

    # Explicit divergence: micro prec (83.3%) != macro prec (80.0%)
    assert micro.confirmed_precision != macro.confirmed_precision


def test_per_recording_speaker_count_and_collapse_detection() -> None:
    """Verifies that P1 shows runaway identities on 2p, P2@2.0 is safe, and P2@2.5 collapses."""
    results = run_full_calibration()

    # P1 has runaway identities on recent-2p (4 confirmed speakers for 2 physical people)
    p1_2p = results["P1"]["per_recording"]["recent-2p"]
    assert len(p1_2p.confirmed_canonical_speakers) == 4

    # P2@2.0 caps recent-2p to exactly 2 confirmed canonical speakers
    p2_20_2p = results["P2@2.0"]["per_recording"]["recent-2p"]
    assert len(p2_20_2p.confirmed_canonical_speakers) == 2
    assert p2_20_2p.confirmed_canonical_speakers == ["Kişi 1", "Kişi 3"]
    assert "Kişi 2" in p2_20_2p.provisional_candidates

    # P2@2.0 preserves 2 confirmed speakers on multi-person-real
    p2_20_multi = results["P2@2.0"]["per_recording"]["multi-person-real"]
    assert len(p2_20_multi.confirmed_canonical_speakers) == 2

    # P2@2.5 suffers pathological collapse on multi-person-real (collapses to 1 confirmed speaker!)
    p2_25_multi = results["P2@2.5"]["per_recording"]["multi-person-real"]
    assert len(p2_25_multi.confirmed_canonical_speakers) == 1
    assert p2_25_multi.confirmed_canonical_speakers == ["Kişi 2"]


def test_deterministic_output() -> None:
    """Running full calibration twice produces identical bit-for-bit results."""
    run1 = run_full_calibration()
    run2 = run_full_calibration()

    for policy_name in run1:
        assert run1[policy_name]["micro"] == run2[policy_name]["micro"]
        assert run1[policy_name]["macro"] == run2[policy_name]["macro"]
        for r_name in run1[policy_name]["per_recording"]:
            m1 = run1[policy_name]["per_recording"][r_name]
            m2 = run2[policy_name]["per_recording"][r_name]
            assert m1.confirmed_precision == m2.confirmed_precision
            assert m1.wrong_confirmed_rate == m2.wrong_confirmed_rate
            assert m1.confirmed_coverage == m2.confirmed_coverage
            assert m1.provisional_rate == m2.provisional_rate
            assert m1.confirmed_canonical_speakers == m2.confirmed_canonical_speakers
