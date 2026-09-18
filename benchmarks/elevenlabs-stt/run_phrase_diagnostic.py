#!/usr/bin/env python3
"""Targeted phrase diagnostic for one human-confirmed Scribe v2 error.

Runs a small fixed matrix of short-excerpt requests against a human-confirmed
reference phrase and writes per-mode comparisons to the private results area.
The reference is a single short sentence used as a targeted regression metric;
no global Turkish accuracy claim is derived from it.

Safety:
- own request guard file with its own budget (default: 4 short-excerpt calls)
- raw provider responses and comparisons land in results/raw and results/private
  (both git-ignored) and are never printed to tracked files
- the API key is read exactly like run_benchmark.py and never logged
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_benchmark as rb  # noqa: E402
from el_parse import group_turns, parse_words  # noqa: E402

RESULTS = SCRIPT_DIR / "results"
PRIVATE = RESULTS / "private"
PHRASE_GUARD_FILE = RESULTS / "phrase-guard.json"
PHRASE_MAX_REQUESTS = 4

EXCERPT = PRIVATE / "ne-yeriz-excerpt.wav"
EXCERPT_NORMALIZED = PRIVATE / "ne-yeriz-excerpt-loudnorm.wav"

HUMAN_REFERENCE = "Ama ne yeriz bilmiyorum. Ayakkabı alabiliriz."

COMMON_FIELDS = {
    "model_id": "scribe_v2",
    "timestamps_granularity": "word",
    "tag_audio_events": "false",
    "no_verbatim": "false",
}

SPECS = {
    "frase-auto": {
        "audio": EXCERPT,
        "fields": {"diarize": "false"},
    },
    "frase-no-verbatim": {
        "audio": EXCERPT,
        "fields": {"language_code": "tur", "diarize": "false", "no_verbatim": "true"},
    },
    "frase-keyterm": {
        "audio": EXCERPT,
        "fields": {"language_code": "tur", "diarize": "false", "keyterms": "ne yeriz"},
    },
    "frase-normalized": {
        "audio": EXCERPT_NORMALIZED,
        "fields": {"language_code": "tur", "diarize": "false"},
    },
}

_TR_LOWER = {"İ": "i", "I": "ı"}


def normalize(text: str) -> str:
    for upper, lower in _TR_LOWER.items():
        text = text.replace(upper, lower)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_ops(reference: str, hypothesis: str) -> dict:
    """Word-level edit distance with substitution/deletion/insertion counts."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    rows, cols = len(ref) + 1, len(hyp) + 1
    distance = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + cost,
            )
    subs = dels = ins = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and distance[i][j] == distance[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "edits": subs + dels + ins,
        "reference_words": len(ref),
        "targeted_wer": round((subs + dels + ins) / len(ref), 4) if ref else None,
    }


def hypothesis_text(payload: dict) -> str:
    words = parse_words(payload)
    return "".join(word.text if index == 0 else f" {word.text}" for index, word in enumerate(words))


def evaluate(label: str, payload: dict, latency: float, audio_seconds: float) -> dict:
    hypothesis = hypothesis_text(payload)
    ops = edit_ops(HUMAN_REFERENCE, hypothesis)
    turns = group_turns(parse_words(payload))
    return {
        "label": label,
        "exact_match": normalize(hypothesis) == normalize(HUMAN_REFERENCE),
        "hypothesis": hypothesis,
        "reference": HUMAN_REFERENCE,
        "language_code": payload.get("language_code"),
        **ops,
        "latency_seconds": round(latency, 3),
        "audio_seconds": audio_seconds,
        "turns": len(turns),
    }


def guarded_request(label: str, key: str) -> dict:
    """Run one request through the shared safe transport with a separate budget."""
    rb.GUARD_FILE = PHRASE_GUARD_FILE
    rb.MAX_REQUESTS = PHRASE_MAX_REQUESTS
    return rb.request(label, key)


def latency_for(label: str) -> float:
    guard = json.loads(PHRASE_GUARD_FILE.read_text(encoding="utf-8"))
    for call in guard.get("calls", []):
        if call.get("label") == label:
            return float(call.get("latency_seconds", 0.0))
    return 0.0


def run(label: str) -> dict:
    rb.CALLS = SPECS
    rb.RESULTS = RESULTS
    key = rb.api_key()
    payload = guarded_request(label, key)
    audio = SPECS[label]["audio"]
    return evaluate(label, payload, latency_for(label), rb.audio_seconds(audio))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=sorted(SPECS), action="append", required=True)
    args = parser.parse_args()

    if not EXCERPT.is_file() or not EXCERPT_NORMALIZED.is_file():
        raise SystemExit(f"private excerpts missing under {PRIVATE}")
    if PHRASE_GUARD_FILE.exists():
        state = json.loads(PHRASE_GUARD_FILE.read_text(encoding="utf-8"))
        used = int(state.get("requests", 0))
        if used + len(args.run) > PHRASE_MAX_REQUESTS:
            raise SystemExit(f"phrase budget exceeded: {used} used, {len(args.run)} requested")

    PRIVATE.mkdir(parents=True, exist_ok=True)
    evaluations = [run(label) for label in args.run]
    target = PRIVATE / "phrase-eval.json"
    existing = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
    existing = [entry for entry in existing if entry["label"] not in args.run] + evaluations
    target.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for entry in evaluations:
        print(
            f"{entry['label']}: exact={entry['exact_match']} "
            f"wer={entry['targeted_wer']} subs={entry['substitutions']} "
            f"dels={entry['deletions']} ins={entry['insertions']} "
            f"latency={entry['latency_seconds']}s audio={entry['audio_seconds']}s",
            flush=True,
        )
        print(f"  hypothesis: {entry['hypothesis']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
