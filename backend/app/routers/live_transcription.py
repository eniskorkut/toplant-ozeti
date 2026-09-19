"""Live ElevenLabs meeting endpoints: ephemeral sessions, rolling speaker windows.

Nothing here exposes provider keys or tokens, stores audio, or keeps embeddings:
a live session is ordering state + canonical speaker timelines + display aliases.
Rolling windows carry raw 16 kHz mono PCM to ElevenLabs from the server side.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi import Path as PathParam
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services.live_sessions import LiveSessionStore, get_live_session_store
from app.services.providers.elevenlabs import transcribe_pcm_window
from app.services.speaker_aliases import (
    AliasValidationError,
    validate_canonical_speaker,
    validate_display_name,
)
from app.services.speaker_matching import Interval
from app.services.transcription import (
    ProviderError,
    ProviderRequestError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/live-transcription", tags=["live-transcription"])

PCM_SAMPLE_RATE = 16_000
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * 2
MIN_WINDOW_SECONDS = 1.0
MAX_WINDOW_SECONDS = 30.0
MAX_PCM_BYTES = int(MAX_WINDOW_SECONDS * PCM_BYTES_PER_SECOND)
SIZE_TOLERANCE_RATIO = 0.1


class LiveSessionCreated(BaseModel):
    live_session_id: str


class SpeakerAssignment(BaseModel):
    canonical_speaker: str
    is_new: bool
    confidence: float | None = None
    evidence: str = "overlap"
    start: float
    end: float
    speech_seconds: float


class SpeakerWindowResult(BaseModel):
    sequence: int
    window: list[float]
    assignments: list[SpeakerAssignment]
    new_speakers: list[str]
    ambiguous_speakers: int = 0
    provider_speakers: int
    latency_seconds: float
    rolling_seconds: float
    label_switches: int


class LiveSpeakerOut(BaseModel):
    canonical_speaker: str
    display_name: str
    speech_seconds: float


class LiveSessionState(BaseModel):
    live_session_id: str
    windows_received: int
    rolling_seconds: float
    label_switches: int
    speakers: list[LiveSpeakerOut]
    aliases: dict[str, str]


class AliasIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=50)


class AliasOut(BaseModel):
    canonical_speaker: str
    display_name: str


def _get_session(store: LiveSessionStore, live_session_id: str):
    session = store.get(live_session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Live session not found or expired"
        )
    return session


@router.post(
    "/sessions", response_model=LiveSessionCreated, status_code=status.HTTP_201_CREATED
)
async def create_live_session(
    store: Annotated[LiveSessionStore, Depends(get_live_session_store)],
) -> LiveSessionCreated:
    session = store.create()
    logger.info("live session %s created", session.id)
    return LiveSessionCreated(live_session_id=session.id)


@router.get("/sessions/{live_session_id}", response_model=LiveSessionState)
async def get_live_session(
    live_session_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    store: Annotated[LiveSessionStore, Depends(get_live_session_store)],
) -> LiveSessionState:
    session = _get_session(store, live_session_id)
    state = session.public_state()
    return LiveSessionState(
        live_session_id=state["live_session_id"],
        windows_received=state["windows_received"],
        rolling_seconds=state["rolling_seconds"],
        label_switches=state["label_switches"],
        speakers=[LiveSpeakerOut(**speaker) for speaker in state["speakers"]],
        aliases=state["aliases"],
    )


@router.delete("/sessions/{live_session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_live_session(
    live_session_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    store: Annotated[LiveSessionStore, Depends(get_live_session_store)],
) -> None:
    store.delete(live_session_id)


@router.post("/sessions/{live_session_id}/speaker-window", response_model=SpeakerWindowResult)
async def submit_speaker_window(
    live_session_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    pcm: Annotated[UploadFile, File(description="16 kHz mono PCM s16le window")],
    start_seconds: Annotated[float, Form(ge=0)],
    end_seconds: Annotated[float, Form(gt=0)],
    sequence: Annotated[int, Form(ge=1)],
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[LiveSessionStore, Depends(get_live_session_store)],
    speaker_count: Annotated[int | None, Form(ge=1, le=12)] = None,
) -> SpeakerWindowResult:
    """Run one rolling diarization window and fold it into the session's timelines."""
    session = _get_session(store, live_session_id)
    if not settings.elevenlabs_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs is not configured.",
        )
    if sequence != session.next_sequence:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Unexpected sequence {sequence}; expected {session.next_sequence}",
        )

    duration = end_seconds - start_seconds
    if duration < MIN_WINDOW_SECONDS or duration > MAX_WINDOW_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Window duration must be between {MIN_WINDOW_SECONDS:g} and "
            f"{MAX_WINDOW_SECONDS:g} seconds",
        )

    try:
        pcm_bytes = await pcm.read()
    finally:
        await pcm.close()

    expected_bytes = int(duration * PCM_BYTES_PER_SECOND)
    tolerated = max(PCM_BYTES_PER_SECOND // 2, int(expected_bytes * SIZE_TOLERANCE_RATIO))
    if len(pcm_bytes) == 0 or len(pcm_bytes) > MAX_PCM_BYTES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="PCM payload size is outside the accepted range",
        )
    if abs(len(pcm_bytes) - expected_bytes) > tolerated:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="PCM payload does not match the declared window duration",
        )

    started = time.perf_counter()
    try:
        words = await asyncio.to_thread(
            transcribe_pcm_window,
            pcm_bytes,
            settings=settings,
            requested_speaker_count=speaker_count,
        )
    except (ProviderUnavailableError, ProviderRequestError) as exc:
        # Rolling diarization must never threaten the recording: report and let the
        # client continue; the sequence stays reserved for a retried window.
        logger.warning("live window %s failed: %s", session.id, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rolling speaker detection is temporarily unavailable",
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Rolling speaker detection returned an unusable response",
        ) from exc
    provider_latency = time.perf_counter() - started

    provider_intervals = _provider_intervals(
        words, offset=start_seconds, window=(start_seconds, end_seconds)
    )
    result = store.apply_window(
        session,
        provider_intervals=provider_intervals,
        window=(start_seconds, end_seconds),
        sequence=sequence,
    )
    session.next_sequence = sequence + 1
    return SpeakerWindowResult(
        sequence=result["sequence"],
        window=result["window"],
        assignments=[SpeakerAssignment(**item) for item in result["assignments"]],
        new_speakers=result["new_speakers"],
        ambiguous_speakers=result["ambiguous_speakers"],
        provider_speakers=len(provider_intervals),
        latency_seconds=round(provider_latency, 3),
        rolling_seconds=round(session.rolling_seconds, 3),
        label_switches=session.label_switches,
    )


def _provider_intervals(
    words, *, offset: float, window: Interval
) -> dict[str, list[Interval]]:
    """Request-local speaker ids -> global-time intervals, clipped to the window."""
    intervals: dict[str, list[Interval]] = {}
    for word in words:
        if word.speaker_id is None:
            continue
        start = max(window[0], offset + word.start)
        end = min(window[1], offset + word.end)
        if end <= start:
            continue
        intervals.setdefault(word.speaker_id, []).append((start, end))
    return intervals


@router.put(
    "/sessions/{live_session_id}/speakers/{canonical_speaker}/alias",
    response_model=AliasOut,
)
async def set_live_alias(
    live_session_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    canonical_speaker: Annotated[str, PathParam(min_length=1, max_length=32)],
    payload: AliasIn,
    store: Annotated[LiveSessionStore, Depends(get_live_session_store)],
) -> AliasOut:
    session = _get_session(store, live_session_id)
    try:
        label = validate_canonical_speaker(canonical_speaker)
        display_name = validate_display_name(payload.display_name)
    except AliasValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    if label not in session.speakers:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canonical speaker is not present in this live session",
        )
    session.aliases[label] = display_name
    return AliasOut(canonical_speaker=label, display_name=display_name)


@router.delete(
    "/sessions/{live_session_id}/speakers/{canonical_speaker}/alias",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def clear_live_alias(
    live_session_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    canonical_speaker: Annotated[str, PathParam(min_length=1, max_length=32)],
    store: Annotated[LiveSessionStore, Depends(get_live_session_store)],
) -> None:
    session = _get_session(store, live_session_id)
    try:
        label = validate_canonical_speaker(canonical_speaker)
    except AliasValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    session.aliases.pop(label, None)
