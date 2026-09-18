"""Local transcription provider: whisper.cpp + TitaNet + the production merge rules.

This wraps the exact production behavior (heuristic timestamps, flash attention ON,
TitaNet threshold 0.80, min_duration_on 0.3 / off 0.5, merge tolerance 0.25 s). It is
the default provider and must keep producing the same transcripts as before.
"""

from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.diarization import DiarizationError, diarize
from app.services.merge import DiarizationSegment as MergeSegment
from app.services.merge import Word, assign_speaker
from app.services.stt import SttError, transcribe
from app.services.transcription import (
    NormalizedWord,
    ProviderError,
    TranscriptionResult,
)

LOCAL_PROVIDER_NAME = "local"
LOCAL_MODEL_NAME = "whisper-large-v3-turbo-q8"


class LocalProvider:
    name = LOCAL_PROVIDER_NAME
    model = LOCAL_MODEL_NAME

    def transcribe(
        self,
        audio_path: Path,
        *,
        requested_speaker_count: int | None,
        settings: Settings,
    ) -> TranscriptionResult:
        try:
            stt_result = transcribe(audio_path, settings)
            diarization_result = diarize(
                audio_path, settings, requested_speaker_count=requested_speaker_count
            )
        except (SttError, DiarizationError) as exc:
            raise ProviderError(str(exc)) from exc

        segments = [
            MergeSegment(segment.start, segment.end, segment.speaker)
            for segment in diarization_result.segments
        ]
        words = [
            NormalizedWord(
                start=word.start,
                end=word.end,
                text=word.text,
                speaker_id=assign_speaker(
                    Word(word.start, word.end, word.text),
                    segments,
                    settings.merge_boundary_tolerance_seconds,
                ),
            )
            for word in stt_result.words
        ]

        return TranscriptionResult(
            text=stt_result.text,
            words=words,
            language=stt_result.language,
            latency_seconds=round(
                stt_result.inference_seconds + diarization_result.inference_seconds, 3
            ),
            provider=LOCAL_PROVIDER_NAME,
            model=LOCAL_MODEL_NAME,
        )
