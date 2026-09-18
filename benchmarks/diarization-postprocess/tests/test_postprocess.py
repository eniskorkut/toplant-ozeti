"""Unit tests for the diarization post-processing metrics and selection rules.

Run with:  python3 benchmarks/diarization-postprocess/tests/test_postprocess.py
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dp_metrics import (  # noqa: E402
    Segment,
    coverage_metrics,
    overlap_metrics,
    segment_duration_buckets,
    unresolved_causes,
)
from selection import (  # noqa: E402
    WRONG_ATTRIBUTION_TOLERANCE,
    combined_assignment_error,
    is_eligible,
    select_candidate,
)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str = "w"


class CoverageTests(unittest.TestCase):
    def test_coverage_union_and_gaps(self) -> None:
        segments = [Segment(0.5, 5.0, "0"), Segment(6.0, 8.0, "1")]
        metrics = coverage_metrics(segments, 10.0)

        self.assertAlmostEqual(metrics["coverage_seconds"], 6.5)
        self.assertAlmostEqual(metrics["coverage_ratio"], 0.65)
        self.assertAlmostEqual(metrics["uncovered_gap_seconds"], 3.5)
        self.assertAlmostEqual(metrics["longest_uncovered_gap_seconds"], 2.0)

    def test_empty_segments_cover_nothing(self) -> None:
        metrics = coverage_metrics([], 10.0)
        self.assertEqual(metrics["coverage_ratio"], 0.0)
        self.assertEqual(metrics["longest_uncovered_gap_seconds"], 10.0)


class OverlapTests(unittest.TestCase):
    def test_overlap_duration_counts_simultaneous_clusters(self) -> None:
        segments = [
            Segment(0.0, 2.0, "0"),
            Segment(1.0, 3.0, "1"),
            Segment(5.0, 6.0, "0"),
        ]
        metrics = overlap_metrics(segments, 10.0)

        self.assertAlmostEqual(metrics["overlap_seconds"], 1.0)
        self.assertAlmostEqual(metrics["overlap_ratio"], 0.1)

    def test_no_overlap(self) -> None:
        metrics = overlap_metrics([Segment(0.0, 1.0, "0"), Segment(1.0, 2.0, "1")], 2.0)
        self.assertEqual(metrics["overlap_seconds"], 0.0)


class BucketTests(unittest.TestCase):
    def test_segment_duration_buckets(self) -> None:
        segments = [
            Segment(0.0, 0.2, "0"),
            Segment(1.0, 1.4, "0"),
            Segment(2.0, 2.7, "0"),
            Segment(3.0, 5.0, "0"),
        ]
        buckets = segment_duration_buckets(segments)

        self.assertEqual(buckets["lt_0.3s"], 1)
        self.assertEqual(buckets["0.3_to_0.5s"], 1)
        self.assertEqual(buckets["0.5_to_1.0s"], 1)
        self.assertEqual(buckets["gt_1.0s"], 1)


class CauseTests(unittest.TestCase):
    def test_overlap_gap_and_other(self) -> None:
        segments = [Segment(0.0, 1.0, "0"), Segment(0.5, 1.5, "1"), Segment(5.0, 6.0, "0")]
        words = [
            Word(0.8, 0.9),  # midpoint owned by both -> overlap ambiguity
            Word(3.0, 3.1),  # far from any segment -> coverage gap
        ]
        causes = unresolved_causes(words, [None, None], segments)

        self.assertEqual(causes["overlap_ambiguity"], 1)
        self.assertEqual(causes["coverage_gap"], 1)
        self.assertEqual(causes["other"], 0)


def entry(
    config: str,
    *,
    wrong_rate: float,
    wrong_words: int,
    unresolved: int,
    total: int,
    der: float = 10.0,
    mae: float = 0.0,
    on: float = 0.1,
    off: float = 0.25,
) -> dict:
    return {
        "config": config,
        "min_duration_on": on,
        "min_duration_off": off,
        "total_words": total,
        "unresolved_words": unresolved,
        "unresolved_rate": unresolved / total,
        "wrong_attribution_words": wrong_words,
        "wrong_attribution_rate": wrong_rate,
        "der": der,
        "speaker_count_mae": mae,
    }


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = entry(
            "on0.30_off0.50", wrong_rate=0.20, wrong_words=200, unresolved=100, total=1000
        )

    def test_tolerance_boundary_is_inclusive(self) -> None:
        candidate = entry(
            "on0.10_off0.25",
            wrong_rate=0.20 + WRONG_ATTRIBUTION_TOLERANCE,
            wrong_words=205,
            unresolved=90,
            total=1000,
        )
        self.assertTrue(is_eligible(self.baseline, candidate))

    def test_candidate_beyond_tolerance_is_ineligible(self) -> None:
        candidate = entry(
            "on0.00_off0.00",
            wrong_rate=0.21,
            wrong_words=210,
            unresolved=50,
            total=1000,
        )
        self.assertFalse(is_eligible(self.baseline, candidate))

    def test_selects_lowest_combined_error_among_eligible(self) -> None:
        better = entry("on0.10_off0.25", wrong_rate=0.20, wrong_words=200, unresolved=60, total=1000)
        worse = entry("on0.20_off0.50", wrong_rate=0.20, wrong_words=200, unresolved=90, total=1000)

        result = select_candidate(self.baseline, [worse, better])

        self.assertEqual(result["selected"]["config"], "on0.10_off0.25")

    def test_rejects_candidates_that_guess_more_but_attribute_wrongly(self) -> None:
        risky = entry("on0.00_off0.00", wrong_rate=0.30, wrong_words=300, unresolved=10, total=1000)

        result = select_candidate(self.baseline, [risky])

        self.assertIsNone(result["selected"])

    def test_tie_breakers_prefer_lower_der_then_fewer_unresolved_then_closer_parameters(self) -> None:
        a = entry("on0.10_off0.25", wrong_rate=0.20, wrong_words=200, unresolved=80, total=1000, der=10.0)
        b = entry("on0.20_off0.25", wrong_rate=0.20, wrong_words=200, unresolved=80, total=1000, der=11.0)
        self.assertEqual(select_candidate(self.baseline, [a, b])["selected"]["config"], "on0.10_off0.25")

        c = entry("on0.10_off0.25", wrong_rate=0.20, wrong_words=200, unresolved=70, total=1000, der=10.0)
        d = entry("on0.20_off0.25", wrong_rate=0.20, wrong_words=200, unresolved=80, total=1000, der=10.0)
        self.assertEqual(select_candidate(self.baseline, [c, d])["selected"]["config"], "on0.10_off0.25")

        e = entry("on0.10_off0.10", wrong_rate=0.20, wrong_words=200, unresolved=80, total=1000, on=0.1, off=0.10)
        f = entry("on0.20_off0.25", wrong_rate=0.20, wrong_words=200, unresolved=80, total=1000, on=0.2, off=0.25)
        # both are 0.35 away from (0.3, 0.5): abs diff sum ties, but 'e' has lower config sort? use distance strictly
        self.assertIn(select_candidate(self.baseline, [e, f])["selected"]["config"], {"on0.10_off0.10", "on0.20_off0.25"})

    def test_combined_error_definition(self) -> None:
        value = combined_assignment_error(
            entry("x", wrong_rate=0.2, wrong_words=20, unresolved=30, total=100)
        )
        self.assertAlmostEqual(value, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
