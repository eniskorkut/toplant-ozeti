"""Benchmark-only sherpa-onnx diarization run (executes inside the backend image).

Reads one 16 kHz mono WAV, runs offline speaker diarization, and emits a single
JSON object on stdout with timings, peak RSS and the segment list.

No preprocessing, no STT, no cluster renaming.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import wave

from pathlib import Path

import numpy as np
import sherpa_onnx


def read_wav_16k_mono(path: str) -> tuple[np.ndarray, int]:
    """Read a PCM16 mono WAV with the standard library; no resampling."""
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)

    if channels != 1 or sample_width != 2:
        raise SystemExit(f"expected mono 16-bit PCM, got channels={channels} width={sample_width}")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--segmentation-model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-clusters", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-on", type=float, default=0.3)
    parser.add_argument("--min-duration-off", type=float, default=0.5)
    parser.add_argument("--label", default="run")
    parser.add_argument("--rttm-out", default=None, help="write system RTTM here")
    parser.add_argument("--rttm-file-id", default=None)
    args = parser.parse_args()

    samples, sample_rate = read_wav_16k_mono(args.audio)

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=args.segmentation_model
            ),
            num_threads=args.threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.embedding_model,
            num_threads=args.threads,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=args.num_clusters,
            threshold=args.threshold,
        ),
        min_duration_on=args.min_duration_on,
        min_duration_off=args.min_duration_off,
    )

    load_started = time.perf_counter()
    diarization = sherpa_onnx.OfflineSpeakerDiarization(config)
    model_load_seconds = time.perf_counter() - load_started

    expected_rate = diarization.sample_rate
    if expected_rate != sample_rate:
        raise SystemExit(f"model expects {expected_rate} Hz, got {sample_rate} Hz")

    started = time.perf_counter()
    result = diarization.process(samples.tolist())
    inference_seconds = time.perf_counter() - started

    segments = result.sort_by_start_time()

    if args.rttm_out:
        if not args.rttm_file_id:
            raise SystemExit("--rttm-file-id is required with --rttm-out")
        lines = [
            "SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> speaker_{speaker} <NA> <NA>".format(
                file_id=args.rttm_file_id,
                start=segment.start,
                duration=segment.end - segment.start,
                speaker=segment.speaker,
            )
            for segment in segments
        ]
        Path(args.rttm_out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "label": args.label,
        "embedding_model": args.embedding_model.rsplit("/", 1)[-1],
        "num_clusters_requested": args.num_clusters,
        "threshold": args.threshold,
        "threads": args.threads,
        "min_duration_on": args.min_duration_on,
        "min_duration_off": args.min_duration_off,
        "audio_seconds": len(samples) / sample_rate,
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "num_speakers": result.num_speakers,
        "num_segments": result.num_segments,
        "segments": [
            {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "speaker": segment.speaker,
            }
            for segment in segments
        ],
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
