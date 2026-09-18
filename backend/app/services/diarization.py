"""Production diarization service: sherpa-onnx 1.10.46 (pyannote 3.0 + TitaNet Small).

No embeddings are returned or persisted — only anonymous cluster ids with time spans.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from app.config import Settings


class DiarizationError(RuntimeError):
    """Raised when diarization fails."""


class DiarizationModelMissingError(DiarizationError):
    """Raised when a diarization model file is unavailable."""


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DiarizationResult:
    segments: list[DiarizationSegment]
    num_speakers: int
    inference_seconds: float
    model_load_seconds: float


def _read_wav_16k_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2:
        raise DiarizationError(
            f"expected 16 kHz mono PCM input, got channels={channels} width={sample_width}"
        )

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_rate


def diarize(
    audio_path: Path,
    settings: Settings,
    *,
    requested_speaker_count: int | None = None,
) -> DiarizationResult:
    segmentation_model = settings.diarization_segmentation_model
    embedding_model = settings.diarization_embedding_model
    for model_path in (segmentation_model, embedding_model):
        if not model_path.exists():
            raise DiarizationModelMissingError(f"diarization model not found: {model_path}")

    samples, sample_rate = _read_wav_16k_mono(audio_path)

    if requested_speaker_count is not None and requested_speaker_count > 0:
        num_clusters = requested_speaker_count
        threshold = settings.diarization_threshold  # bypassed when num_clusters is set
    else:
        num_clusters = -1
        threshold = settings.diarization_threshold

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation_model)
            ),
            num_threads=settings.diarization_threads,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding_model),
            num_threads=settings.diarization_threads,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=threshold,
        ),
        min_duration_on=settings.diarization_min_duration_on,
        min_duration_off=settings.diarization_min_duration_off,
    )

    load_started = time.perf_counter()
    try:
        diarization = sherpa_onnx.OfflineSpeakerDiarization(config)
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        raise DiarizationError(f"failed to initialise diarization: {exc}") from exc
    model_load_seconds = time.perf_counter() - load_started

    if diarization.sample_rate != sample_rate:
        raise DiarizationError(
            f"model expects {diarization.sample_rate} Hz but the file is {sample_rate} Hz"
        )

    started = time.perf_counter()
    try:
        result = diarization.process(samples.tolist())
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        raise DiarizationError(f"diarization failed: {exc}") from exc
    inference_seconds = time.perf_counter() - started

    segments = [
        DiarizationSegment(
            start=segment.start,
            end=segment.end,
            speaker=str(segment.speaker),
        )
        for segment in result.sort_by_start_time()
    ]

    return DiarizationResult(
        segments=segments,
        num_speakers=result.num_speakers,
        inference_seconds=round(inference_seconds, 3),
        model_load_seconds=round(model_load_seconds, 3),
    )
