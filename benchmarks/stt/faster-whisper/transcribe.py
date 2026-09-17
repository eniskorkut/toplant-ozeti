"""Benchmark-only faster-whisper transcription (sequential or batched).

Emits one JSON object on stdout.

Documented behavior difference of the official batched mode
(`BatchedInferencePipeline`, faster-whisper 1.2.1):

- With `vad_filter=False`, the batched API **requires** explicit clip boundaries
  for audio longer than `chunk_length` (it raises RuntimeError otherwise). Fixed
  30 s windows are therefore passed explicitly, which is the same window length
  the pipeline uses internally when VAD is enabled.
- Each clip is decoded independently, so cross-window context
  (`condition_on_previous_text`) is lost in batched mode; this cannot be avoided
  with the official API.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
import wave


def wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def fixed_clip_timestamps(duration_seconds: float, clip_seconds: float) -> list[dict[str, float]]:
    clips: list[dict[str, float]] = []
    start = 0.0
    while start < duration_seconds:
        end = min(start + clip_seconds, duration_seconds)
        clips.append({"start": round(start, 3), "end": round(end, 3)})
        start = end
    return clips


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="tr")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=1, help="1 = sequential (official default)")
    parser.add_argument("--clip-seconds", type=float, default=30.0)
    parser.add_argument("--vad", action="store_true", help="off by default")
    args = parser.parse_args()

    from faster_whisper import BatchedInferencePipeline, WhisperModel

    load_started = time.perf_counter()
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type=args.compute_type,
        cpu_threads=args.threads,
    )
    load_seconds = time.perf_counter() - load_started

    duration_seconds = wav_duration_seconds(args.audio)
    common = {
        "language": args.language,
        "task": "transcribe",
        "beam_size": args.beam_size,
        "vad_filter": args.vad,
    }

    clip_count: int | None = None
    if args.batch_size > 1:
        clips = None
        if not args.vad:
            clips = fixed_clip_timestamps(duration_seconds, args.clip_seconds)
            clip_count = len(clips)
        pipeline = BatchedInferencePipeline(model=model)
        started = time.perf_counter()
        segments, info = pipeline.transcribe(
            args.audio,
            batch_size=args.batch_size,
            clip_timestamps=clips,
            condition_on_previous_text=True,
            **common,
        )
        transcript = "".join(segment.text for segment in segments).strip()
        inference_seconds = time.perf_counter() - started
        mode = "batched"
    else:
        started = time.perf_counter()
        segments, info = model.transcribe(
            args.audio,
            condition_on_previous_text=True,
            word_timestamps=False,
            **common,
        )
        transcript = "".join(segment.text for segment in segments).strip()
        inference_seconds = time.perf_counter() - started
        mode = "sequential"

    print(
        json.dumps(
            {
                "engine": "faster-whisper",
                "engine_version": importlib.metadata.version("faster-whisper"),
                "model": args.model,
                "compute_type": args.compute_type,
                "mode": mode,
                "batch_size": args.batch_size,
                "clip_seconds": args.clip_seconds if args.batch_size > 1 and not args.vad else None,
                "clip_count": clip_count,
                "threads": args.threads,
                "beam_size": args.beam_size,
                "vad": args.vad,
                "language": info.language,
                "language_probability": info.language_probability,
                "audio_duration_seconds": info.duration or duration_seconds,
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
