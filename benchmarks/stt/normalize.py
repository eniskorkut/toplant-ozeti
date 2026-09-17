"""Benchmark-only Turkish text normalization and Word Error Rate.

Standard library only: no NLP dependency is added for scoring.

Turkish letters (ç ğ ı i ö ş ü) are preserved: only punctuation is removed,
never diacritics.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path

PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Unicode-normalize, lowercase, drop punctuation, collapse whitespace."""
    normalized = unicodedata.normalize("NFC", text).lower()
    without_punctuation = PUNCTUATION_RE.sub(" ", normalized)
    return WHITESPACE_RE.sub(" ", without_punctuation).strip()


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split(" ") if normalized else []


def word_error_counts(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via edit-distance DP."""
    reference_length = len(reference)
    hypothesis_length = len(hypothesis)

    distances = [[0] * (hypothesis_length + 1) for _ in range(reference_length + 1)]
    for i in range(reference_length + 1):
        distances[i][0] = i
    for j in range(hypothesis_length + 1):
        distances[0][j] = j

    for i in range(1, reference_length + 1):
        for j in range(1, hypothesis_length + 1):
            substitution_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j - 1] + substitution_cost,
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
            )

    substitutions = deletions = insertions = 0
    i, j = reference_length, hypothesis_length
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitution_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if distances[i][j] == distances[i - 1][j - 1] + substitution_cost:
                substitutions += substitution_cost
                i -= 1
                j -= 1
                continue
        if i > 0 and distances[i][j] == distances[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        insertions += 1
        j -= 1

    return substitutions, deletions, insertions


def score(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Full WER report for one hypothesis against the reference text."""
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    substitutions, deletions, insertions = word_error_counts(reference_tokens, hypothesis_tokens)
    errors = substitutions + deletions + insertions
    reference_words = len(reference_tokens)

    return {
        "wer": (errors / reference_words) if reference_words else float("nan"),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "reference_words": reference_words,
        "hypothesis_words": len(hypothesis_tokens),
        "normalized_hypothesis": normalize_text(hypothesis),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: normalize.py <reference_file> <hypothesis_file>", file=sys.stderr)
        return 2

    reference = Path(argv[1]).read_text(encoding="utf-8")
    hypothesis = Path(argv[2]).read_text(encoding="utf-8")
    result = score(reference, hypothesis)

    for key in ("wer", "substitutions", "deletions", "insertions", "reference_words"):
        print(f"{key}: {result[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
