"""Non-network unit tests for the ElevenLabs benchmark layer.

Run with:  python3 benchmarks/elevenlabs-stt/tests/test_elevenlabs.py
No test performs a network call.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_benchmark  # noqa: E402
from el_parse import (  # noqa: E402
    ElevenError,
    ElevenRequestGuard,
    UNKNOWN_SPEAKER,
    group_turns,
    parse_words,
    redact_error,
    response_metrics,
    speaker_metrics,
    timestamp_health,
)

SAMPLE = {
    "language_code": "tur",
    "language_probability": 0.98,
    "words": [
        {"type": "word", "text": "Merhaba", "start": 0.0, "end": 0.5, "speaker_id": "speaker_0"},
        {"type": "spacing", "text": " ", "start": 0.5, "end": 0.5},
        {"type": "word", "text": "nasılsın", "start": 0.5, "end": 1.0, "speaker_id": "speaker_0"},
        {"type": "audio_event", "text": "(gülüşme)", "start": 1.0, "end": 1.2},
        {"type": "word", "text": "iyiyim", "start": 1.2, "end": 1.6, "speaker_id": "speaker_1"},
        {"type": "word", "text": "bilinmeyen", "start": 1.6, "end": 1.9},
    ],
}


class ParseTests(unittest.TestCase):
    def test_only_word_entries_are_parsed(self) -> None:
        words = parse_words(SAMPLE)
        self.assertEqual([word.text for word in words], ["Merhaba", "nasılsın", "iyiyim", "bilinmeyen"])

    def test_missing_speaker_id_stays_unassigned(self) -> None:
        words = parse_words(SAMPLE)
        self.assertIsNone(words[-1].speaker)
        self.assertIsNotNone(words[0].speaker)

    def test_empty_text_is_skipped(self) -> None:
        payload = {"words": [{"type": "word", "text": "   ", "start": 0, "end": 0, "speaker_id": "s"}]}
        self.assertEqual(parse_words(payload), [])

    def test_empty_payload(self) -> None:
        self.assertEqual(parse_words({}), [])


class GroupingTests(unittest.TestCase):
    def test_consecutive_same_speaker_words_form_one_turn(self) -> None:
        turns = group_turns(parse_words(SAMPLE))
        self.assertEqual([turn.speaker for turn in turns], ["speaker_0", "speaker_1", UNKNOWN_SPEAKER])
        self.assertEqual(turns[0].text, "Merhaba nasılsın")
        self.assertEqual(turns[0].words, 2)

    def test_unassigned_words_keep_their_own_turn(self) -> None:
        turns = group_turns(parse_words(SAMPLE))
        self.assertEqual(turns[-1].speaker, UNKNOWN_SPEAKER)
        self.assertEqual(turns[-1].text, "bilinmeyen")

    def test_unassigned_between_same_speaker_does_not_merge(self) -> None:
        payload = {
            "words": [
                {"type": "word", "text": "a", "start": 0.0, "end": 0.3, "speaker_id": "s0"},
                {"type": "word", "text": "b", "start": 0.4, "end": 0.6},
                {"type": "word", "text": "c", "start": 0.7, "end": 1.0, "speaker_id": "s0"},
            ]
        }
        turns = group_turns(parse_words(payload))
        self.assertEqual([turn.speaker for turn in turns], ["s0", UNKNOWN_SPEAKER, "s0"])


class MetricTests(unittest.TestCase):
    def test_speaker_metrics_counts(self) -> None:
        metrics = speaker_metrics(parse_words(SAMPLE))
        self.assertEqual(metrics["words"], 4)
        self.assertEqual(metrics["speaker_tagged_words"], 3)
        self.assertEqual(metrics["untagged_words"], 1)
        self.assertEqual(metrics["distinct_speakers"], 2)
        self.assertEqual(metrics["speaker_turns"], 2)
        self.assertEqual(metrics["words_per_speaker"], {"speaker_0": 2, "speaker_1": 1})
        self.assertEqual(metrics["first_word_seconds"], 0.0)
        self.assertEqual(metrics["last_word_seconds"], 1.9)

    def test_response_metrics_does_not_include_text(self) -> None:
        metrics = response_metrics(SAMPLE)
        serialized = json.dumps(metrics)
        self.assertNotIn("Merhaba", serialized)
        self.assertNotIn("nasılsın", serialized)
        self.assertEqual(metrics["text_chars"], len("Merhaba nasılsın iyiyim bilinmeyen"))

    def test_timestamp_health(self) -> None:
        payload = {
            "words": [
                {"type": "word", "text": "a", "start": 1.0, "end": 1.5},
                {"type": "word", "text": "b", "start": 1.4, "end": 1.4},
            ]
        }
        health = timestamp_health(parse_words(payload))
        self.assertEqual(health["zero_duration_words"], 1)
        self.assertGreaterEqual(health["monotonicity_errors"], 1)

    def test_rapid_flip_detection(self) -> None:
        payload = {
            "words": [
                {"type": "word", "text": "a", "start": 0.0, "end": 0.4, "speaker_id": "s0"},
                {"type": "word", "text": "b", "start": 0.5, "end": 0.7, "speaker_id": "s1"},
                {"type": "word", "text": "c", "start": 0.8, "end": 1.2, "speaker_id": "s0"},
            ]
        }
        self.assertEqual(speaker_metrics(parse_words(payload))["rapid_flips"], 1)


class GuardTests(unittest.TestCase):
    def test_request_budget_is_enforced(self) -> None:
        guard = ElevenRequestGuard(maximum=2)
        guard.check()
        guard.record(label="a", audio_seconds=1.0, latency_seconds=0.1, status=200)
        guard.record(label="b", audio_seconds=2.0, latency_seconds=0.2, status=200)
        with self.assertRaises(ElevenError):
            guard.check()
        summary = guard.summary()
        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["total_audio_seconds_sent"], 3.0)

    def test_budget_default_is_six(self) -> None:
        self.assertEqual(ElevenRequestGuard().maximum, 6)


class EnvLocalTests(unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(run_benchmark.read_env_local(Path("/nonexistent/.env.local")), {})

    def test_parses_values_and_ignores_comments(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.local"
            path.write_text(
                "# comment\nELEVENLABS_API_KEY='sk_test_value'\nOTHER=plain\n",
                encoding="utf-8",
            )
            values = run_benchmark.read_env_local(path)
        self.assertEqual(values["ELEVENLABS_API_KEY"], "sk_test_value")
        self.assertEqual(values["OTHER"], "plain")

    def test_default_env_local_path_is_inside_the_harness(self) -> None:
        self.assertEqual(run_benchmark.ENV_LOCAL.name, ".env.local")
        self.assertEqual(run_benchmark.ENV_LOCAL.parent.name, "elevenlabs-stt")


class AggregateSafetyTests(unittest.TestCase):
    def test_metrics_summary_never_contains_response_words(self) -> None:
        # Simulates the aggregate rule: only counts/metrics may leave the harness.
        metrics = response_metrics(SAMPLE)
        serialized = json.dumps(metrics, ensure_ascii=False)
        for word in ("Merhaba", "nasılsın", "iyiyim", "bilinmeyen"):
            self.assertNotIn(word, serialized)

    def test_guard_summary_has_no_key_or_headers(self) -> None:
        guard = ElevenRequestGuard(maximum=6)
        guard.record(label="real-known4", audio_seconds=42.84, latency_seconds=4.2, status=200)
        serialized = json.dumps(guard.summary())
        self.assertNotIn("xi-api-key", serialized)
        self.assertNotIn("authorization", serialized.lower())
        for forbidden in ("/Users/", "/data/", ".wav"):
            self.assertNotIn(forbidden, serialized)


class RedactionTests(unittest.TestCase):
    def test_error_redaction_removes_secrets(self) -> None:
        secret = "sk_test_secret_value"
        message = f"request failed with xi-api-key {secret} header"
        redacted = redact_error(message, [secret])
        self.assertNotIn(secret, redacted)
        self.assertIn("<redacted>", redacted)

    def test_redaction_is_bounded(self) -> None:
        self.assertLessEqual(len(redact_error("x" * 1000, [])), 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
