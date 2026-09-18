"""Non-network unit tests for the targeted phrase diagnostic.

Run with:  python3 benchmarks/elevenlabs-stt/tests/test_phrase_eval.py
No test performs a network call.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_phrase_diagnostic as diagnostic  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_strips_punctuation_and_lowercases(self) -> None:
        self.assertEqual(
            diagnostic.normalize("Ama ne yeriz bilmiyorum. Ayakkabı alabiliriz."),
            "ama ne yeriz bilmiyorum ayakkabı alabiliriz",
        )

    def test_turkish_dotted_and_dotless_i(self) -> None:
        self.assertEqual(diagnostic.normalize("İSTANBUL IĞDIR"), "istanbul ığdır")


class EditOpsTests(unittest.TestCase):
    def test_exact_match_has_zero_edits(self) -> None:
        ops = diagnostic.edit_ops(diagnostic.HUMAN_REFERENCE, diagnostic.HUMAN_REFERENCE)
        self.assertEqual(ops["edits"], 0)
        self.assertEqual(ops["targeted_wer"], 0.0)

    def test_known_failure_ne_yeriz_to_neyi(self) -> None:
        ops = diagnostic.edit_ops(
            diagnostic.HUMAN_REFERENCE,
            "Ama neyi bilmiyorum. Ayakkabı alabiliriz.",
        )
        self.assertEqual(ops["reference_words"], 6)
        self.assertEqual(ops["edits"], 2)
        self.assertEqual(ops["targeted_wer"], 0.3333)

    def test_deletion_and_insertion_classes(self) -> None:
        deletion = diagnostic.edit_ops("bir iki üç", "bir iki")
        self.assertEqual((deletion["deletions"], deletion["insertions"], deletion["substitutions"]), (1, 0, 0))
        insertion = diagnostic.edit_ops("bir iki", "bir iki üç")
        self.assertEqual((insertion["deletions"], insertion["insertions"], insertion["substitutions"]), (0, 1, 0))

    def test_empty_reference_is_none_wer(self) -> None:
        self.assertIsNone(diagnostic.edit_ops("", "bir şey")["targeted_wer"])


class SpecTests(unittest.TestCase):
    def test_matrix_is_exactly_four_modes(self) -> None:
        self.assertEqual(
            sorted(diagnostic.SPECS),
            ["frase-auto", "frase-keyterm", "frase-no-verbatim", "frase-normalized"],
        )

    def test_auto_mode_omits_language_code(self) -> None:
        self.assertNotIn("language_code", diagnostic.SPECS["frase-auto"]["fields"])

    def test_only_no_verbatim_mode_enables_no_verbatim(self) -> None:
        enabled = [
            label
            for label, spec in diagnostic.SPECS.items()
            if spec["fields"].get("no_verbatim") == "true"
        ]
        self.assertEqual(enabled, ["frase-no-verbatim"])

    def test_only_keyterm_mode_uses_keyterms(self) -> None:
        with_keyterms = [
            label for label, spec in diagnostic.SPECS.items() if "keyterms" in spec["fields"]
        ]
        self.assertEqual(with_keyterms, ["frase-keyterm"])

    def test_normalized_mode_uses_normalized_audio(self) -> None:
        self.assertEqual(
            diagnostic.SPECS["frase-normalized"]["audio"], diagnostic.EXCERPT_NORMALIZED
        )

    def test_budget_is_four_short_calls(self) -> None:
        self.assertEqual(diagnostic.PHRASE_MAX_REQUESTS, 4)


if __name__ == "__main__":
    unittest.main()
