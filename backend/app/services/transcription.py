"""Provider-independent transcription results and turn formation.

The application layer only consumes `TranscriptionResult`: a full text plus words with
optional anonymous speaker ids. Local and cloud providers keep their own internals; the
shared helpers below normalize the result and form speaker turns exactly the way the
local merge has always done it (Kişi N by first appearance, unresolved breaks turns).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.models import UNKNOWN_SPEAKER
from app.services.merge import SpeakerTurn


class ProviderError(RuntimeError):
    """Base class for transcription provider failures."""


class ProviderConfigurationError(ProviderError):
    """Raised when the selected provider is not configured correctly."""


class ProviderRequestError(ProviderError):
    """Raised when the provider rejects the request (auth, quota, malformed input)."""


class ProviderUnavailableError(ProviderError):
    """Raised for transient provider problems (timeout, connection, 5xx)."""


class ProviderResponseError(ProviderError):
    """Raised when the provider response cannot be parsed or validated."""


@dataclass(frozen=True)
class NormalizedWord:
    start: float
    end: float
    text: str
    speaker_id: str | None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    words: list[NormalizedWord]
    language: str | None
    latency_seconds: float
    provider: str
    model: str


class TranscriptionProvider(Protocol):
    name: str
    model: str

    def transcribe(
        self,
        audio_path: Path,
        *,
        requested_speaker_count: int | None,
        settings: Settings,
    ) -> TranscriptionResult:
        ...


def label_speakers(cluster_ids: list[str]) -> dict[str, str]:
    """Session-local labels by first appearance (raw provider ids are never exposed)."""
    labels: dict[str, str] = {}
    for cluster_id in cluster_ids:
        if cluster_id not in labels:
            labels[cluster_id] = f"Kişi {len(labels) + 1}"
    return labels


def form_turns(words: list[NormalizedWord]) -> list[SpeakerTurn]:
    """Group consecutive words of the same speaker; unresolved words break turns.

    This mirrors the local merge semantics exactly: speaker ids are mapped to Kişi N by
    first appearance and a missing speaker becomes Bilinmeyen without inheriting a
    neighbouring speaker.
    """
    labels = label_speakers([word.speaker_id for word in words if word.speaker_id is not None])

    turns: list[SpeakerTurn] = []
    previous_label: str | None = None
    for word in words:
        label = labels.get(word.speaker_id, UNKNOWN_SPEAKER) if word.speaker_id else UNKNOWN_SPEAKER
        if turns and previous_label == label:
            turn = turns[-1]
            turn.end = word.end
            turn.text = f"{turn.text} {word.text}".strip()
            turn.words += 1
        else:
            turns.append(
                SpeakerTurn(
                    speaker=label,
                    start=word.start,
                    end=word.end,
                    text=word.text,
                    words=1,
                )
            )
        previous_label = label
    return turns


def build_provider(settings: Settings) -> TranscriptionProvider:
    """Select the configured provider; the local stack remains the default."""
    from app.services.providers.elevenlabs import ElevenLabsProvider
    from app.services.providers.local import LocalProvider

    provider = settings.transcription_provider.strip().lower()
    if provider == "local":
        return LocalProvider()
    if provider == "elevenlabs":
        return ElevenLabsProvider(settings)
    raise ProviderConfigurationError(
        f"unknown transcription provider {settings.transcription_provider!r} "
        "(expected 'local' or 'elevenlabs')"
    )
