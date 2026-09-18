"""Unit tests for the benchmark-only word/speaker merge (standard library only).

Run with:  python3 benchmarks/merge/tests/test_merge.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge import (  # noqa: E402
    DiarizationSegment,
    Word,
    assign_speaker,
    merge,
    timestamp_health,
)


class AssignSpeakerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = [
            DiarizationSegment(0.0, 2.0, "speaker_0"),
            DiarizationSegment(2.0, 4.0, "speaker_1"),
        ]

    def test_maximum_overlap_wins(self) -> None:
        # overlaps: speaker_0 -> 0.2 s, speaker_1 -> 0.6 s
        word = Word(1.8, 2.6, "x")
        self.assertEqual(assign_speaker(word, self.segments), "speaker_1")

    def test_straddling_word_goes_to_longer_overlap(self) -> None:
        # overlaps: speaker_0 -> 0.5 s, speaker_1 -> 0.4 s
        word = Word(1.5, 2.4, "x")
        self.assertEqual(assign_speaker(word, self.segments), "speaker_0")

    def test_zero_duration_word_uses_midpoint(self) -> None:
        word = Word(1.6, 1.6, "x")
        self.assertEqual(assign_speaker(word, self.segments), "speaker_0")

    def test_small_gap_within_tolerance(self) -> None:
        word = Word(4.1, 4.2, "x")
        self.assertEqual(assign_speaker(word, self.segments, tolerance=0.25), "speaker_1")

    def test_gap_beyond_tolerance_is_unresolved(self) -> None:
        word = Word(4.5, 4.6, "x")
        self.assertIsNone(assign_speaker(word, self.segments, tolerance=0.25))

    def test_no_segments_is_unresolved(self) -> None:
        self.assertIsNone(assign_speaker(Word(0.0, 1.0, "x"), []))


class TurnFormationTests(unittest.TestCase):
    def test_consecutive_same_speaker_forms_one_turn(self) -> None:
        words = [Word(0.0, 0.5, "a"), Word(0.5, 1.0, "b"), Word(1.0, 1.5, "c")]
        segments = [DiarizationSegment(0.0, 2.0, "speaker_0")]
        result = merge(words, segments, audio_seconds=2.0)

        self.assertEqual(len(result["turns"]), 1)
        self.assertEqual(result["turns"][0]["text"], "a b c")
        self.assertEqual(result["unresolved_words"], 0)

    def test_short_turn_is_preserved_not_absorbed(self) -> None:
        words = [
            Word(0.0, 0.5, "Evet"),
            Word(2.0, 2.25, "Tamam"),
            Word(4.0, 4.5, "devam"),
        ]
        segments = [
            DiarizationSegment(0.0, 1.0, "speaker_0"),
            DiarizationSegment(1.9, 2.4, "speaker_1"),
            DiarizationSegment(3.9, 5.0, "speaker_0"),
        ]
        result = merge(words, segments, audio_seconds=5.0)

        self.assertEqual([turn["speaker"] for turn in result["turns"]], ["speaker_0", "speaker_1", "speaker_0"])
        self.assertEqual(result["turns"][1]["text"], "Tamam")
        self.assertGreaterEqual(result["short_turns_preserved"], 1)

    def test_unresolved_words_are_not_attributed(self) -> None:
        words = [Word(0.0, 0.5, "a"), Word(10.0, 10.5, "b")]
        segments = [DiarizationSegment(0.0, 1.0, "speaker_0")]
        result = merge(words, segments, audio_seconds=11.0)

        self.assertEqual(result["unresolved_words"], 1)
        self.assertEqual(len(result["turns"]), 1)
        self.assertEqual(result["turns"][0]["text"], "a")

    def test_rapid_flip_detection(self) -> None:
        words = [
            Word(0.0, 0.4, "a"),
            Word(0.5, 0.7, "b"),
            Word(0.8, 1.2, "c"),
        ]
        segments = [
            DiarizationSegment(0.0, 0.45, "speaker_0"),
            DiarizationSegment(0.5, 0.75, "speaker_1"),
            DiarizationSegment(0.8, 1.5, "speaker_0"),
        ]
        result = merge(words, segments, audio_seconds=1.5)
        self.assertEqual(result["rapid_flips"], 1)


class InvariantTests(unittest.TestCase):
    def test_clean_merge_has_no_violations(self) -> None:
        words = [Word(0.0, 0.5, "a"), Word(0.6, 1.0, "b")]
        segments = [DiarizationSegment(0.0, 1.0, "speaker_0")]
        result = merge(words, segments, audio_seconds=1.0)
        self.assertEqual(result["structural_violations"], [])

    def test_monotonicity_violation_is_detected(self) -> None:
        words = [Word(0.0, 1.0, "a"), Word(0.5, 0.8, "b")]
        health = timestamp_health(words, audio_seconds=1.0)
        self.assertGreater(health["monotonicity_errors"], 0)

    def test_zero_duration_and_overlap_counters(self) -> None:
        words = [Word(0.0, 0.0, "a"), Word(0.0, 0.5, "b")]
        health = timestamp_health(words, audio_seconds=1.0)
        self.assertEqual(health["zero_duration_words"], 1)
        self.assertGreaterEqual(health["overlapping_word_timestamps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
