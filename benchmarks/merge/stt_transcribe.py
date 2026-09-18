"""Benchmark-only STT timestamp extraction (runs inside the STT benchmark images).

Emits unified JSON for both engines:

    whisper.cpp    -> `-ojf` full JSON, word-piece tokens with millisecond offsets
                      aggregated into words at space boundaries
    faster-whisper -> word_timestamps=True, native word intervals

No application dependency, no model changes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import resource
import subprocess
import time
import wave
from pathlib import Path

WHISPER_OUT_PREFIX = "/tmp/merge_stt_out"


def wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def run_whisper_cpp(args: argparse.Namespace) -> dict:
    command = [
        "whisper-cli",
        "-m", args.model,
        "-f", args.audio,
        "-l", args.language,
        "-t", str(args.threads),
        "-bs", str(args.beam_size),
        "-ojf",
        "-of", WHISPER_OUT_PREFIX,
        "-np",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    inference_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(f"whisper-cli failed: {completed.stderr[-2000:]}")

    data = json.loads(Path(f"{WHISPER_OUT_PREFIX}.json").read_text(encoding="utf-8"))

    words: list[dict] = []
    segments: list[dict] = []
    current: dict | None = None
    for segment in data["transcription"]:
        segments.append(
            {
                "start": segment["offsets"]["from"] / 1000.0,
                "end": segment["offsets"]["to"] / 1000.0,
                "text": segment["text"].strip(),
            }
        )
        for token in segment.get("tokens", []):
            text = token["text"]
            if text.startswith("[_"):
                continue
            start = token["offsets"]["from"] / 1000.0
            end = token["offsets"]["to"] / 1000.0
            if text.startswith(" ") or current is None:
                if current is not None:
                    words.append(current)
                current = {"start": start, "end": end, "text": text.strip(), "dtw": token.get("t_dtw", -1)}
            else:
                current["end"] = end
                current["text"] = f"{current['text']}{text.strip()}"

    if current is not None:
        words.append(current)

    return {
        "engine": "whisper.cpp",
        "engine_version": Path("/opt/whisper.cpp/VERSION").read_text(encoding="utf-8").strip(),
        "timestamp_source": "token offsets from -ojf JSON (heuristic token timestamps; t_dtw=-1 means DTW was not computed)",
        "model": Path(args.model).name,
        "words": words,
        "segments": segments,
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "inference_seconds": round(inference_seconds, 3),
        "model_load_seconds": None,
        # whisper-cli runs as a child process: the model lives in the child, so the
        # peak RSS of the children is the meaningful number here.
        "peak_rss_kb": max(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    }


def run_faster_whisper(args: argparse.Namespace) -> dict:
    from faster_whisper import WhisperModel

    load_started = time.perf_counter()
    model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=args.threads)
    model_load_seconds = time.perf_counter() - load_started

    started = time.perf_counter()
    iterator, _ = model.transcribe(
        args.audio,
        language=args.language,
        task="transcribe",
        beam_size=args.beam_size,
        vad_filter=False,
        condition_on_previous_text=True,
        word_timestamps=True,
    )
    segments: list[dict] = []
    words: list[dict] = []
    for segment in iterator:
        segments.append(
            {"start": segment.start, "end": segment.end, "text": segment.text.strip()}
        )
        for word in segment.words or []:
            words.append(
                {
                    "start": word.start,
                    "end": word.end,
                    "text": word.word.strip(),
                    "probability": round(word.probability, 4),
                }
            )
    inference_seconds = time.perf_counter() - started

    return {
        "engine": "faster-whisper",
        "engine_version": importlib.metadata.version("faster-whisper"),
        "timestamp_source": "native word_timestamps=True intervals",
        "model": args.model,
        "words": words,
        "segments": segments,
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "inference_seconds": round(inference_seconds, 3),
        "model_load_seconds": round(model_load_seconds, 3),
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["whisper-cpp", "faster-whisper"], required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="tr")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    payload = run_whisper_cpp(args) if args.engine == "whisper-cpp" else run_faster_whisper(args)
    payload["audio_seconds"] = round(wav_duration_seconds(args.audio), 3)
    payload["language"] = args.language
    payload["threads"] = args.threads
    payload["beam_size"] = args.beam_size
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
