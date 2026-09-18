"""Benchmark-only STT runner with optional DTW token timestamps (runs in the whisper image).

Builds word intervals from whisper.cpp token output:
- `heuristic`: token `offsets` (the production default)
- `dtw`: token `t_dtw` (centiseconds) from `-dtw large.v3.turbo`

DTW requires flash attention to be disabled in our v1.9.4 build
(`dtw_token_timestamps is not supported with flash_attn - disabling`), so the DTW mode
also passes `-nfa`. Decoding flags are otherwise identical.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

OUT_PREFIX = "/tmp/overlap_stt"


def run(args: argparse.Namespace) -> dict:
    command = [
        "whisper-cli",
        "-m", args.model,
        "-f", args.audio,
        "-l", args.language,
        "-t", str(args.threads),
        "-bs", str(args.beam_size),
        "-ojf",
        "-of", OUT_PREFIX,
        "-np",
    ]
    if args.timestamps in ("dtw", "heuristic_nfa"):
        # DTW is incompatible with flash attention in v1.9.4; the control mode runs
        # the same no-flash-attention decoding without DTW.
        command.append("-nfa")
    if args.timestamps == "dtw":
        command += ["-dtw", "large.v3.turbo"]

    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(f"whisper-cli failed: {completed.stderr[-1500:]}")

    payload = json.loads(Path(f"{OUT_PREFIX}.json").read_text(encoding="utf-8"))
    words: list[dict] = []
    current: dict | None = None
    for segment in payload.get("transcription", []):
        for token in segment.get("tokens", []):
            text = token["text"]
            if text.startswith("[_"):
                continue
            offsets = token["offsets"]
            dtw_seconds = token.get("t_dtw", -1) / 100.0 if token.get("t_dtw", -1) >= 0 else None
            if text.startswith(" ") or current is None:
                if current is not None:
                    words.append(current)
                current = {
                    "text": text.strip(),
                    "offset_start": offsets["from"] / 1000.0,
                    "offset_end": offsets["to"] / 1000.0,
                    "dtw_points": [dtw_seconds] if dtw_seconds is not None else [],
                }
            else:
                current["text"] += text.strip()
                current["offset_end"] = offsets["to"] / 1000.0
                if dtw_seconds is not None:
                    current["dtw_points"].append(dtw_seconds)
        if current is not None:
            words.append(current)
            current = None

    for word in words:
        points = [point for point in word["dtw_points"] if point is not None]
        if args.timestamps == "dtw" and points:
            word["start"] = min(points)
            word["end"] = max(points)
        else:
            word["start"] = word["offset_start"]
            word["end"] = word["offset_end"]
        word["dtw_start"] = min(points) if points else None
        word["dtw_end"] = max(points) if points else None
        del word["dtw_points"]

    text = " ".join(segment["text"].strip() for segment in payload.get("transcription", [])).strip()
    return {
        "timestamps": args.timestamps,
        "language": args.language,
        "model": Path(args.model).name,
        "text": text,
        "words": words,
        "word_count": len(words),
        "decoding_seconds": round(elapsed, 3),
        "flash_attention_disabled": args.timestamps in ("heuristic_nfa", "dtw"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="tr")
    parser.add_argument(
        "--timestamps", choices=["heuristic", "heuristic_nfa", "dtw"], default="heuristic"
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = run(args)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(
        f"{args.timestamps}: {result['word_count']} words in {result['decoding_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
