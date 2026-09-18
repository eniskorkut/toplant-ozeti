"""ElevenLabs Scribe v2 provider (opt-in cloud path).

Direct HTTP, no SDK. The API key is read from configuration (a SecretStr), used only in
the `xi-api-key` header and never logged, persisted or included in errors.

Response handling: only `type == "word"` items become words; a missing `speaker_id`
stays absent (never invented); timestamps are validated before the application sees
them. Failures are raised as typed errors — there is no silent fallback to the local
pipeline.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import httpx

from app.config import Settings
from app.services.transcription import (
    NormalizedWord,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderUnavailableError,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
AUDIO_DURATION_TOLERANCE_SECONDS = 1.0


class ElevenLabsProvider:
    name = "elevenlabs"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.elevenlabs_stt_model

    def transcribe(
        self,
        audio_path: Path,
        *,
        requested_speaker_count: int | None,
        settings: Settings,
    ) -> TranscriptionResult:
        secret = settings.elevenlabs_api_key
        api_key = secret.get_secret_value() if secret is not None else ""
        if not api_key:
            raise ProviderConfigurationError(
                "ELEVENLABS_API_KEY is not configured for the elevenlabs provider"
            )

        data = self.build_form_data(
            settings, requested_speaker_count=requested_speaker_count
        )
        audio_seconds = _audio_duration_seconds(audio_path)

        import time

        started = time.perf_counter()
        try:
            with audio_path.open("rb") as handle:
                response = httpx.post(
                    API_URL,
                    headers={"xi-api-key": api_key},
                    data=data,
                    files={"file": (audio_path.name, handle, "audio/wav")},
                    timeout=settings.elevenlabs_timeout_seconds,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning("elevenlabs request failed: %s", type(exc).__name__)
            raise ProviderUnavailableError(
                f"ElevenLabs connection problem: {type(exc).__name__}"
            ) from exc
        latency = time.perf_counter() - started

        self._raise_for_status(response)
        payload = self._parse_envelope(response)

        words = self.parse_words(payload)
        if not words:
            raise ProviderResponseError("ElevenLabs response contained no spoken words")
        self.validate_timestamps(words, audio_seconds=audio_seconds)

        text = str(payload.get("text") or "").strip() or " ".join(word.text for word in words)
        return TranscriptionResult(
            text=text,
            words=words,
            language=payload.get("language_code"),
            latency_seconds=round(latency, 3),
            provider=self.name,
            model=self.model,
        )

    # --- request -----------------------------------------------------------

    @staticmethod
    def build_form_data(
        settings: Settings, *, requested_speaker_count: int | None
    ) -> dict[str, str]:
        """Form fields for one request (public so tests can assert exact behavior)."""
        data = {
            "model_id": settings.elevenlabs_stt_model,
            "timestamps_granularity": "word",
            "diarize": "true",
            "tag_audio_events": "false",
            "no_verbatim": "false",
            "language_code": settings.elevenlabs_language_code,
        }
        if requested_speaker_count:
            # ElevenLabs treats num_speakers as the maximum expected speaker count.
            data["num_speakers"] = str(requested_speaker_count)
        elif settings.elevenlabs_diarization_threshold is not None:
            # diarization_threshold is only valid without num_speakers.
            data["diarization_threshold"] = str(settings.elevenlabs_diarization_threshold)
        return data

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        status = response.status_code
        if status == 200:
            return
        if status in (401, 403):
            raise ProviderRequestError(
                f"ElevenLabs rejected the request (HTTP {status}); check the API key"
            )
        if status == 429:
            raise ProviderRequestError("ElevenLabs quota or rate limit reached (HTTP 429)")
        if status >= 500:
            raise ProviderUnavailableError(f"ElevenLabs is unavailable (HTTP {status})")
        raise ProviderRequestError(f"ElevenLabs request failed (HTTP {status})")

    @staticmethod
    def _parse_envelope(response: httpx.Response) -> dict:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderResponseError("ElevenLabs response was not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ProviderResponseError("ElevenLabs response was not an object")
        return payload

    # --- response ----------------------------------------------------------

    @staticmethod
    def parse_words(payload: dict) -> list[NormalizedWord]:
        entries = payload.get("words")
        if entries is None:
            raise ProviderResponseError("ElevenLabs response is missing the words list")
        if not isinstance(entries, list):
            raise ProviderResponseError("ElevenLabs words list is malformed")

        words: list[NormalizedWord] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "word":
                continue
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            try:
                start = float(entry.get("start", 0.0))
                end = float(entry.get("end", 0.0))
            except (TypeError, ValueError) as exc:
                raise ProviderResponseError("ElevenLabs word timestamps are malformed") from exc
            speaker = entry.get("speaker_id")
            words.append(
                NormalizedWord(
                    start=start,
                    end=end,
                    text=text,
                    speaker_id=str(speaker) if speaker not in (None, "") else None,
                )
            )
        return words

    @staticmethod
    def validate_timestamps(words: list[NormalizedWord], *, audio_seconds: float) -> None:
        previous_end = 0.0
        for index, word in enumerate(words):
            if word.start < -1e-9 or word.end < 0:
                raise ProviderResponseError(f"word {index} has a negative timestamp")
            if word.start > word.end:
                raise ProviderResponseError(f"word {index} has start > end")
            if word.start < previous_end - 1e-6:
                raise ProviderResponseError(f"word {index} breaks timestamp monotonicity")
            if word.end > audio_seconds + AUDIO_DURATION_TOLERANCE_SECONDS:
                raise ProviderResponseError(f"word {index} ends beyond the audio duration")
            previous_end = max(previous_end, word.end)


def _audio_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()
