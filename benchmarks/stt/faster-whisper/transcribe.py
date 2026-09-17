"""Benchmark-only faster-whisper transcription. Emits one JSON object on stdout."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    load_started = time.perf_counter()
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
        cpu_threads=args.threads,
    )
    load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    segments, info = model.transcribe(
        args.audio,
        language=args.language,
        task="transcribe",
        beam_size=args.beam_size,
        vad_filter=False,
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    transcript = "".join(segment.text for segment in segments).strip()
    inference_seconds = time.perf_counter() - started

    print(
        json.dumps(
            {
                "engine": "faster-whisper",
                "engine_version": importlib.metadata.version("faster-whisper"),
                "model": args.model,
                "compute_type": args.compute_type,
                "threads": args.threads,
                "beam_size": args.beam_size,
                "vad": False,
                "language": info.language,
                "language_probability": info.language_probability,
                "audio_duration_seconds": info.duration,
                "model_load_seconds": round(load_seconds, 3),
                "inference_seconds": round(inference_seconds, 3),
                "transcript": transcript,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
