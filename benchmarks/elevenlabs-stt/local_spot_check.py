#!/usr/bin/env python3
"""Per-file local comparison for whmpa/wjhgf, reusing stored sweep segments.

Runs inside the backend image; performs no diarization or STT inference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/overlap")

from evaluate import evaluate_assignment, parse_rttm, speaker_metrics  # noqa: E402

from app.services.merge import DiarizationSegment, Word, assign_speaker  # noqa: E402

FILES = ["whmpa", "wjhgf"]
CONFIGS = ["on0.30_off0.50", "on0.00_off0.00"]
SEGMENTS = Path("/sweep-segments")
STT_RAW = Path("/stt-raw")
RTTM = Path("/voxconverse/voxconverse/dev")


def main() -> int:
    output: dict = {}
    for file_id in FILES:
        words_payload = json.loads((STT_RAW / f"{file_id}--heuristic.json").read_text(encoding="utf-8"))
        words = [Word(word["start"], word["end"], word["text"]) for word in words_payload["words"]]
        reference = parse_rttm(RTTM / f"{file_id}.rttm")
        output[file_id] = {"reference_speakers": len({turn.speaker for turn in reference})}
        for config in CONFIGS:
            payload = json.loads((SEGMENTS / config / f"{file_id}.json").read_text(encoding="utf-8"))
            segments = [
                DiarizationSegment(segment["start"], segment["end"], f"speaker_{segment['speaker']}")
                for segment in payload["segments"]
            ]
            speakers = [assign_speaker(word, segments, 0.25) for word in words]
            evaluation = evaluate_assignment(words, speakers, reference)
            metrics = speaker_metrics(words, speakers)
            output[file_id][config] = {
                "detected_speakers": payload["num_speakers"],
                "segments": len(segments),
                "agreement": evaluation["agreement"],
                "wrong_attribution_rate": evaluation["wrong_attribution_rate"],
                "wrong_attribution_words": evaluation["wrong_attribution_words"],
                "unresolved_words": evaluation["unresolved_words"],
                "unresolved_rate": evaluation["unresolved_rate"],
                "compared_words": evaluation["compared_words"],
                "turns": metrics["turns"],
                "rapid_flips": metrics["rapid_flips"],
            }
    Path("/out/local-spot-check.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
