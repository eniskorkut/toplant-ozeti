"""Unit tests for the diagnostic metric helpers (standard library only).

Run with:  python3 benchmarks/real-meeting-diagnostic/tests/test_metrics.py
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diagnostic_metrics import (  # noqa: E402
    diarization_metrics,
    merge_metrics,
    normalize_words,
    segment_coverage,
    short_examples,
    text_delta,
    unresolved_causes,
    unresolved_word_gaps,
)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: str


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str = ""
    words: int = 1


class NormalizeTests(unittest.TestCase):
    def test_preserves_turkish_letters_and_strips_punctuation(self) -> None:
        self.assertEqual(
            normalize_words("Bugünkü toplantı, yarın!"),
            ["bugünkü", "toplantı", "yarın"],
        )


class TextDeltaTests(unittest.TestCase):
    def test_identical_text_has_zero_difference(self) -> None:
        words = ["bir", "iki", "üç"]
        delta = text_delta(words, list(words))
        self.assertEqual(delta["differing_word_positions"], 0)
        self.assertEqual(delta["similarity_ratio"], 1.0)

    def test_reports_differing_positions(self) -> None:
        delta = text_delta(["bir", "iki"], ["bir", "üç"])
        self.assertGreater(delta["differing_word_positions"], 0)

    def test_examples_are_short_and_limited(self) -> None:
        auto = [f"a{index}" for index in range(40)]
        tr = [f"b{index}" for index in range(40)]
        examples = short_examples(auto, tr, max_examples=3)
        self.assertLessEqual(len(examples), 3)
        for example in examples:
            self.assertLessEqual(len(example["auto_snippet"]), 160)


class CoverageTests(unittest.TestCase):
    def test_coverage_union_and_longest_gap(self) -> None:
        segments = [Segment(0.0, 10.0, "0"), Segment(11.0, 20.0, "1")]
        coverage = segment_coverage(segments, 25.0)
        self.assertAlmostEqual(coverage["coverage_seconds"], 19.0)
        self.assertAlmostEqual(coverage["longest_gap_seconds"], 5.0)

    def test_empty_segments(self) -> None:
        coverage = segment_coverage([], 10.0)
        self.assertEqual(coverage["coverage_ratio"], 0.0)


class UnresolvedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [Segment(0.0, 5.0, "0"), Segment(5.0, 10.0, "1")]

    def test_gap_buckets(self) -> None:
        words = [Word(2.0, 2.5, "a"), Word(12.0, 12.5, "b"), Word(30.0, 30.5, "c")]
        gaps = unresolved_word_gaps(words, self.segments)
        self.assertEqual(gaps["count"], 3)
        self.assertEqual(gaps["buckets"]["overlapping_or_within_tolerance"], 1)
        self.assertEqual(gaps["buckets"]["one_to_3s"], 1)
        self.assertEqual(gaps["buckets"]["over_3s"], 1)

    def test_causes_split_overlap_and_coverage(self) -> None:
        overlapping = [Segment(4.0, 6.0, "0"), Segment(4.5, 6.5, "1")]
        words = [
            Word(5.0, 5.5, "overlap"),  # midpoint owned by both
            Word(20.0, 20.5, "gap"),  # far from every segment
            Word(4.9, 5.1, "boundary"),  # near/touching exactly one segment
        ]
        causes = unresolved_causes(words, overlapping + [Segment(4.8, 5.2, "2")], tolerance=0.25)
        self.assertEqual(causes["total"], 3)
        self.assertGreaterEqual(causes["overlap_region"], 1)
        self.assertGreaterEqual(causes["coverage_gap"], 1)


class MergeMetricsTests(unittest.TestCase):
    def test_counts_speakers_turns_and_unknowns(self) -> None:
        turns = [
            Turn("Kişi 1", 0.0, 2.0, "bir iki", words=2),
            Turn("Bilinmeyen", 2.0, 2.3, "???", words=1),
            Turn("Kişi 2", 2.3, 5.0, "üç", words=1),
        ]
        metrics = merge_metrics(turns, 5.0, real_speakers=4)
        self.assertEqual(metrics["final_speaker_count"], 2)
        self.assertEqual(metrics["turn_count"], 3)
        self.assertEqual(metrics["unknown_turn_count"], 1)
        self.assertEqual(metrics["speaker_changes"], 2)
        self.assertEqual(metrics["speaker_count_error"], -2)
        self.assertEqual(metrics["per_speaker_words"]["Bilinmeyen"], 1)

    def test_rapid_flip_detection(self) -> None:
        turns = [
            Turn("Kişi 1", 0.0, 1.0),
            Turn("Kişi 2", 1.0, 1.2),
            Turn("Kişi 1", 1.2, 3.0),
        ]
        self.assertEqual(merge_metrics(turns, 3.0, real_speakers=None)["rapid_flips"], 1)


class DiarizationMetricsTests(unittest.TestCase):
    def test_non_empty_clusters(self) -> None:
        segments = [
            Segment(0.0, 1.0, "0"),
            Segment(1.0, 2.0, "0"),
            Segment(2.0, 3.0, "1"),
        ]
        metrics = diarization_metrics(segments, 3.0)
        self.assertEqual(metrics["non_empty_clusters"], 2)
        self.assertEqual(metrics["segment_count"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
