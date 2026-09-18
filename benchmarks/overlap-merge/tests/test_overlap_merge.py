"""Unit tests for the benchmark-only context resolver and evaluation helpers.

Run with:  python3 benchmarks/overlap-merge/tests/test_overlap_merge.py
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_resolver import (  # noqa: E402
    Segment,
    apply_resolver,
    bridge_coverage_gaps,
    nearest_activity_distance,
    resolve_contextual,
    speaker_support,
)
from evaluate import evaluate_assignment, speaker_metrics  # noqa: E402
import run_stt  # noqa: E402


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str = "w"


class SpeakerSupportTests(unittest.TestCase):
    def test_support_sums_active_duration_over_the_window(self) -> None:
        segments = [
            Segment(0.0, 1.0, "a"),
            Segment(0.5, 2.0, "b"),
            Segment(1.5, 3.0, "b"),
        ]
        support = speaker_support(segments, 0.6, 1.8)
        self.assertAlmostEqual(support["a"], 0.4)
        # b is active in [0.5, 2.0] and [1.5, 3.0]: 1.2 s + 0.3 s inside the window
        self.assertAlmostEqual(support["b"], 1.5)

    def test_nearest_activity_distance(self) -> None:
        segments = [Segment(0.0, 1.0, "a")]
        self.assertEqual(nearest_activity_distance(0.5, 0.6, segments), 0.0)
        self.assertAlmostEqual(nearest_activity_distance(1.2, 1.3, segments), 0.2)
        self.assertEqual(nearest_activity_distance(5.0, 5.1, []), float("inf"))


class ContextResolverTests(unittest.TestCase):
    def setUp(self) -> None:
        # Two speakers overlap between 4.5 and 6.0; speaker "a" dominates the window.
        self.segments = [
            Segment(4.0, 5.6, "a"),
            Segment(5.0, 5.4, "b"),
            Segment(8.0, 9.0, "b"),
        ]
        self.words = [Word(5.0, 5.1, "x"), Word(7.0, 7.1, "y")]

    def test_unique_highest_support_resolves(self) -> None:
        resolved, stats = resolve_contextual(
            self.words, [None, None], self.segments, radius=0.5, margin=0.10
        )
        self.assertEqual(resolved[0], "a")  # 0.6 s support vs 0.1 s for b
        self.assertEqual(stats["resolved"], 1)

    def test_margin_rejects_close_support(self) -> None:
        resolved, stats = resolve_contextual(
            self.words, [None, None], self.segments, radius=0.5, margin=0.60
        )
        self.assertIsNone(resolved[0])
        self.assertEqual(stats["rejected_margin"], 1)

    def test_no_support_stays_unresolved(self) -> None:
        resolved, _ = resolve_contextual(
            self.words, [None, None], self.segments, radius=0.25, margin=0.10
        )
        self.assertIsNone(resolved[1])

    def test_exact_tie_is_never_broken_by_ordering(self) -> None:
        segments = [Segment(0.0, 1.0, "a"), Segment(0.0, 1.0, "b")]
        resolved, stats = resolve_contextual(
            [Word(0.5, 0.5, "x")], [None], segments, radius=0.5, margin=0.10
        )
        self.assertIsNone(resolved[0])
        self.assertEqual(stats["rejected_tie"], 1)


class BridgeResolverTests(unittest.TestCase):
    def test_bridges_short_gap_between_equal_speakers(self) -> None:
        segments = [Segment(0.0, 1.0, "a"), Segment(1.4, 3.0, "a")]
        words = [Word(0.2, 0.4), Word(1.1, 1.2), Word(2.0, 2.2)]
        speakers = ["a", None, "a"]

        resolved, stats = bridge_coverage_gaps(words, speakers, segments, max_gap=0.5)

        self.assertEqual(resolved[1], "a")
        self.assertEqual(stats["resolved"], 1)

    def test_does_not_bridge_large_gap(self) -> None:
        segments = [Segment(0.0, 1.0, "a"), Segment(3.0, 4.0, "a")]
        words = [Word(0.2, 0.4), Word(2.0, 2.1), Word(3.2, 3.4)]
        speakers = ["a", None, "a"]

        resolved, stats = bridge_coverage_gaps(words, speakers, segments, max_gap=0.5)

        self.assertIsNone(resolved[1])
        self.assertEqual(stats["resolved"], 0)

    def test_does_not_bridge_different_speakers(self) -> None:
        segments = [Segment(0.0, 1.0, "a"), Segment(1.2, 2.0, "b")]
        words = [Word(0.2, 0.4), Word(1.05, 1.1), Word(1.5, 1.6)]
        speakers = ["a", None, "b"]

        resolved, _ = bridge_coverage_gaps(words, speakers, segments, max_gap=0.5)

        self.assertIsNone(resolved[1])

    def test_short_word_does_not_inherit_neighbour_without_activity(self) -> None:
        segments = [Segment(0.0, 0.5, "a")]
        words = [Word(0.1, 0.2), Word(5.0, 5.1)]
        speakers = ["a", None]

        resolved, _ = bridge_coverage_gaps(words, speakers, segments, max_gap=0.5)

        self.assertIsNone(resolved[1])


class ApplyResolverTests(unittest.TestCase):
    def test_context_then_bridge_and_stats(self) -> None:
        segments = [
            Segment(0.0, 1.0, "a"),
            Segment(1.05, 1.5, "a"),
            Segment(1.5, 3.0, "b"),
        ]
        words = [Word(0.2, 0.3), Word(1.1, 1.2), Word(2.0, 2.1)]
        baseline = ["a", None, None]

        resolved, stats = apply_resolver(
            words, baseline, segments, radius=0.25, margin=0.10, bridge_max_gap=0.5
        )

        self.assertIsNotNone(resolved[1])
        self.assertEqual(stats["baseline_unresolved"], 2)
        self.assertEqual(
            stats["resolved_by_context"] + stats["resolved_by_bridge"], 2
        )
        self.assertEqual(stats["remaining_unresolved"], 0)


class EvaluationTests(unittest.TestCase):
    def test_agreement_uses_best_permutation(self) -> None:
        # cluster_b speaks during spk2's turn, cluster_a during spk1's turn
        words = [
            Word(0.0, 0.5),
            Word(0.6, 1.0),
            Word(2.0, 2.5),
            Word(3.0, 3.5),
        ]
        speakers = ["cluster_b", "cluster_b", "cluster_a", "cluster_a"]
        reference = [
            Segment(0.0, 1.0, "spk2"),
            Segment(2.0, 4.0, "spk1"),
        ]

        evaluation = evaluate_assignment(words, speakers, reference)

        self.assertEqual(evaluation["compared_words"], 4)
        self.assertEqual(evaluation["agreement"], 1.0)
        self.assertEqual(evaluation["wrong_attribution_words"], 0)

    def test_wrong_attribution_is_counted(self) -> None:
        words = [Word(0.0, 0.5), Word(1.0, 1.5)]
        speakers = ["cluster_a", "cluster_a"]
        reference = [Segment(0.0, 1.0, "spk2"), Segment(1.0, 2.0, "spk1")]

        evaluation = evaluate_assignment(words, speakers, reference)

        self.assertEqual(evaluation["compared_words"], 2)
        self.assertGreater(evaluation["wrong_attribution_words"], 0)

    def test_unresolved_rate_and_newly_resolved(self) -> None:
        words = [Word(0.0, 0.5), Word(1.0, 1.5)]
        reference = [Segment(0.0, 1.0, "spk1"), Segment(1.0, 2.0, "spk1")]

        evaluation = evaluate_assignment(
            words, ["a", None], reference, baseline_speakers=[None, None]
        )

        self.assertEqual(evaluation["unresolved_words"], 1)
        self.assertEqual(evaluation["unresolved_rate"], 0.5)
        self.assertEqual(evaluation["newly_resolved_words"], 1)

    def test_speaker_metrics_turns_and_flips(self) -> None:
        words = [Word(0.0, 0.4), Word(0.5, 0.7), Word(0.8, 1.2)]
        speakers = ["a", "b", "a"]

        metrics = speaker_metrics(words, speakers)

        self.assertEqual(metrics["turns"], 3)
        self.assertEqual(metrics["rapid_flips"], 1)
        self.assertEqual(metrics["speaker_changes"], 2)



class ExtractionLanguageTests(unittest.TestCase):
    """The orchestration layer assigns languages explicitly; VoxConverse is English."""

    def plan(self, kind: str) -> list[dict]:
        return [job for job in run_stt.extraction_plan() if job["kind"] == kind]

    def test_voxconverse_extraction_uses_english(self) -> None:
        jobs = self.plan("voxconverse")
        self.assertEqual(len(jobs), 12)
        self.assertEqual({job["language"] for job in jobs}, {"en"})

    def test_real_meeting_extraction_uses_turkish(self) -> None:
        jobs = self.plan("real")
        self.assertEqual([job["language"] for job in jobs], ["tr"])

    def test_controlled_turkish_recordings_use_turkish(self) -> None:
        jobs = self.plan("turkish_controlled")
        self.assertEqual({job["language"] for job in jobs}, {"tr"})

    def test_every_job_declares_a_language(self) -> None:
        for job in run_stt.extraction_plan():
            self.assertIn(job["language"], {"en", "tr"})


class SttModeMetadataTests(unittest.TestCase):
    def test_flash_attention_flag_matches_the_mode(self) -> None:
        # -nfa is required for DTW in our v1.9.4 build; the control mode keeps the
        # metadata consistent so the aggregate can not silently report the wrong value.
        self.assertFalse("heuristic" in ("heuristic_nfa", "dtw"))
        self.assertTrue("heuristic_nfa" in ("heuristic_nfa", "dtw"))
        self.assertTrue("dtw" in ("heuristic_nfa", "dtw"))
if __name__ == "__main__":
    unittest.main(verbosity=2)
