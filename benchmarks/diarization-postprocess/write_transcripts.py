#!/usr/bin/env python3
"""Write git-ignored local transcripts for the swept real-meeting configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/overlap")

from app.services.merge import DiarizationSegment, Word, assign_speaker, label_speakers  # noqa: E402

MEETING = "b1095740120b4b1e96db337da961ea63"
RESULTS = Path("/bench/results")
PRIVATE = RESULTS / "private"
STT_RAW = Path("/stt-raw")


def main() -> int:
    payload = json.loads((STT_RAW / f"{MEETING}--heuristic.json").read_text(encoding="utf-8"))
    words = [Word(word["start"], word["end"], word["text"]) for word in payload["words"]]
    PRIVATE.mkdir(parents=True, exist_ok=True)

    for config_dir in sorted((RESULTS / "segments" / "real").iterdir()):
        segments_payload = json.loads((config_dir / f"{MEETING}.json").read_text(encoding="utf-8"))
        segments = [
            DiarizationSegment(segment["start"], segment["end"], f"speaker_{segment['speaker']}")
            for segment in segments_payload["segments"]
        ]
        speakers = [assign_speaker(word, segments, 0.25) for word in words]
        labels = label_speakers([speaker for speaker in speakers if speaker is not None])

        lines: list[list] = []
        for word, speaker in zip(words, speakers, strict=True):
            label = labels.get(speaker, "Bilinmeyen") if speaker is not None else "Bilinmeyen"
            if lines and lines[-1][0] == label:
                lines[-1][1] = f"{lines[-1][1]} {word.text}"
                lines[-1][3] = word.end
            else:
                lines.append([label, word.text, word.start, word.end])
        target = PRIVATE / f"real-{config_dir.name}.txt"
        target.write_text(
            "\n".join(f"[{entry[2]:8.3f}] {entry[0]}: {entry[1]}" for entry in lines) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {target} ({len(lines)} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
