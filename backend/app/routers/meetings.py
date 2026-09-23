"""Meeting processing API: queue processing, poll status, fetch the transcript."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import delete, exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    UNKNOWN_SPEAKER,
    Meeting,
    MeetingAnalysis,
    MeetingChatMessage,
    TranscriptTurn,
)
from app.services.analysis_pipeline import payload_from_row
from app.services.llm_provider import LlmConfigurationError, LlmProviderError, build_provider
from app.services.meeting_chat import (
    answer_meeting_question,
    clear_chat_messages,
    list_chat_messages,
)
from app.services.speaker_aliases import (
    AliasValidationError,
    clear_alias,
    list_aliases,
    set_alias,
    speaker_labels_in_transcript,
    validate_canonical_speaker,
    validate_display_name,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/meetings", tags=["meetings"])

MAX_SPEAKER_COUNT = 12


def _iso_utc(value: datetime) -> str:
    """UTC ISO-8601 with an explicit offset so clients render local time.

    SQLite returns naive datetimes for timezone-aware columns; those values are
    UTC by construction (written with datetime.now(UTC)). Without the offset a
    browser would interpret them as local time (three hours off in Türkiye).
    """
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.isoformat()


class ProcessRequest(BaseModel):
    speaker_count: int | None = Field(
        default=None,
        ge=1,
        le=MAX_SPEAKER_COUNT,
        description="Optional known speaker count; null selects automatic clustering.",
    )
    transcription_provider: Literal["local", "elevenlabs"] | None = Field(
        default=None,
        description="Per-meeting provider override; null uses the configured default.",
    )


class MeetingStatus(BaseModel):
    meeting_id: str
    status: str
    created_at: str
    duration_seconds: float | None
    requested_speaker_count: int | None
    processing_error: str | None
    has_transcript: bool
    # Safe provider metadata only (never keys, config internals or raw responses).
    transcription_provider: str | None = None
    transcription_model: str | None = None


class MeetingSummary(BaseModel):
    meeting_id: str
    created_at: str
    duration_seconds: float | None
    status: str
    requested_speaker_count: int | None
    has_transcript: bool
    analysis_status: str | None
    transcription_provider: str | None = None
    transcription_model: str | None = None


class MeetingList(BaseModel):
    meetings: list[MeetingSummary]
    count: int


class KeyPointOut(BaseModel):
    text: str
    source_turn_ordinals: list[int]
    timestamp_seconds: float


class DecisionOut(BaseModel):
    text: str
    source_turn_ordinals: list[int]
    timestamp_seconds: float


class ActionItemOut(BaseModel):
    task: str
    owner: str | None
    due_date_text: str | None
    source_turn_ordinals: list[int]
    timestamp_seconds: float


class ImportantMomentOut(BaseModel):
    title: str
    description: str
    source_turn_ordinal: int
    timestamp_seconds: float


class AnalysisResponse(BaseModel):
    meeting_id: str
    status: str
    provider: str | None = None
    model: str | None = None
    summary: str | None = None
    key_points: list[KeyPointOut] = []
    topics: list[str] = []
    decisions: list[DecisionOut] = []
    action_items: list[ActionItemOut] = []
    important_moments: list[ImportantMomentOut] = []
    analysis_error: str | None = None
    input_chars: int | None = None
    latency_seconds: float | None = None
    repair_attempts: int = 0


class TranscriptTurnOut(BaseModel):
    ordinal: int
    speaker: str
    start_seconds: float
    end_seconds: float
    text: str


class TranscriptResponse(BaseModel):
    meeting_id: str
    status: str
    duration_seconds: float | None
    speakers: list[str]
    unresolved_label: str
    unresolved_turns: int
    turns: list[TranscriptTurnOut]


def _meeting_status(meeting: Meeting, has_transcript: bool) -> MeetingStatus:
    return MeetingStatus(
        meeting_id=meeting.id,
        status=meeting.status,
        created_at=_iso_utc(meeting.created_at),
        duration_seconds=meeting.duration_seconds,
        requested_speaker_count=meeting.requested_speaker_count,
        processing_error=meeting.processing_error,
        has_transcript=has_transcript,
        transcription_provider=meeting.transcription_provider,
        transcription_model=meeting.transcription_model,
    )


async def _get_meeting(session: AsyncSession, meeting_id: str) -> Meeting:
    meeting = await session.get(Meeting, meeting_id)
    if meeting is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting not found")
    return meeting


async def _turn_count(session: AsyncSession, meeting_id: str) -> int:
    turns = (
        await session.execute(
            select(TranscriptTurn.id).where(TranscriptTurn.meeting_id == meeting_id).limit(1)
        )
    ).scalars()
    return len(list(turns))


@router.post(
    "/{meeting_id}/process",
    response_model=MeetingStatus,
    status_code=status.HTTP_202_ACCEPTED,
)
async def queue_processing(
    request: ProcessRequest,
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> MeetingStatus:
    """Queue a meeting for processing. Idempotent: no duplicate work is created."""
    meeting = await _get_meeting(session, meeting_id)

    if meeting.status in (
        MEETING_STATUS_COMPLETED,
        MEETING_STATUS_PROCESSING,
        MEETING_STATUS_QUEUED,
    ):
        # Already done or already in flight: report the current state unchanged.
        return _meeting_status(meeting, await _turn_count(session, meeting_id) > 0)

    if request.transcription_provider == "elevenlabs" and not settings.elevenlabs_configured:
        # Fail before queueing: no silent fallback, no half-configured cloud job.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs transcription is not configured.",
        )

    meeting.status = MEETING_STATUS_QUEUED
    meeting.requested_speaker_count = request.speaker_count
    meeting.requested_transcription_provider = request.transcription_provider
    meeting.processing_error = None
    await session.commit()
    await session.refresh(meeting)
    logger.info("meeting %s queued (speaker_count=%s)", meeting.id, request.speaker_count)
    return _meeting_status(meeting, False)


@router.get("", response_model=MeetingList)
async def list_meetings(
    session: Annotated[AsyncSession, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MeetingList:
    """Newest first. Transcript text is never included in the list."""
    has_turns = exists(
        select(TranscriptTurn.id).where(TranscriptTurn.meeting_id == Meeting.id)
    )
    rows = (
        await session.execute(
            select(Meeting, has_turns, MeetingAnalysis.status)
            .outerjoin(MeetingAnalysis, MeetingAnalysis.meeting_id == Meeting.id)
            .order_by(Meeting.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()

    meetings = [
        MeetingSummary(
            meeting_id=meeting.id,
            created_at=_iso_utc(meeting.created_at),
            duration_seconds=meeting.duration_seconds,
            status=meeting.status,
            requested_speaker_count=meeting.requested_speaker_count,
            has_transcript=bool(turns),
            analysis_status=analysis_status,
            transcription_provider=meeting.transcription_provider,
            transcription_model=meeting.transcription_model,
        )
        for meeting, turns, analysis_status in rows
    ]
    return MeetingList(meetings=meetings, count=len(meetings))


class AliasIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=50)


class MeetingSpeakersOut(BaseModel):
    meeting_id: str
    speakers: list[str]
    aliases: dict[str, str]


@router.get("/{meeting_id}/speakers", response_model=MeetingSpeakersOut)
async def get_meeting_speakers(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> MeetingSpeakersOut:
    """Canonical speaker labels of this meeting plus its display aliases."""
    await _get_meeting(session, meeting_id)
    labels = await speaker_labels_in_transcript(session, meeting_id)
    aliases = await list_aliases(session, meeting_id)
    return MeetingSpeakersOut(
        meeting_id=meeting_id,
        speakers=sorted(labels | set(aliases), key=_speaker_sort_key),
        aliases=aliases,
    )


@router.put("/{meeting_id}/speakers/{canonical_speaker}/alias", response_model=dict)
async def set_meeting_speaker_alias(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    canonical_speaker: Annotated[str, PathParam(min_length=1, max_length=32)],
    payload: AliasIn,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    await _get_meeting(session, meeting_id)
    try:
        label = validate_canonical_speaker(canonical_speaker)
        display_name = validate_display_name(payload.display_name)
    except AliasValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    existing_labels = await speaker_labels_in_transcript(session, meeting_id)
    aliases = await list_aliases(session, meeting_id)
    if existing_labels and label not in existing_labels and label not in aliases:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Canonical speaker is not present in this meeting",
        )

    await set_alias(
        session, meeting_id=meeting_id, canonical_speaker=label, display_name=display_name
    )
    return {"canonical_speaker": label, "display_name": display_name}


@router.delete(
    "/{meeting_id}/speakers/{canonical_speaker}/alias",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def reset_meeting_speaker_alias(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    canonical_speaker: Annotated[str, PathParam(min_length=1, max_length=32)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    await _get_meeting(session, meeting_id)
    try:
        label = validate_canonical_speaker(canonical_speaker)
    except AliasValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    await clear_alias(session, meeting_id=meeting_id, canonical_speaker=label)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _speaker_sort_key(label: str) -> tuple[int, str]:
    if label.startswith("Kişi "):
        try:
            return (int(label.split(" ", 1)[1]), label)
        except ValueError:
            return (999, label)
    return (1000, label)


@router.delete("/{meeting_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_meeting(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Delete a meeting with its transcript and analysis.

    Safety rules:
    - queued/processing meetings (and meetings with an active analysis) are
      refused with 409 so no worker writes into a deleted row
    - artifacts are only removed after the DB commit and only when no other
      meeting still references the same relative path (comparison meetings
      intentionally share the source audio)
    - every filesystem target is resolved from the persisted relative path and
      must stay inside the configured data directory
    - filesystem cleanup is best effort: a failure never breaks DB consistency
    """
    meeting = await _get_meeting(session, meeting_id)

    if meeting.status in (MEETING_STATUS_QUEUED, MEETING_STATUS_PROCESSING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Meeting is being processed and cannot be deleted.",
        )

    analysis_status = (
        await session.execute(
            select(MeetingAnalysis.status).where(MeetingAnalysis.meeting_id == meeting_id)
        )
    ).scalar_one_or_none()
    if analysis_status in (ANALYSIS_STATUS_QUEUED, ANALYSIS_STATUS_PROCESSING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Meeting analysis is running and cannot be deleted.",
        )

    artifacts = {meeting.audio_mp3_path, meeting.processing_wav_path} - {None, ""}

    await session.execute(delete(TranscriptTurn).where(TranscriptTurn.meeting_id == meeting_id))
    await session.execute(delete(MeetingAnalysis).where(MeetingAnalysis.meeting_id == meeting_id))
    await session.execute(
        delete(MeetingChatMessage).where(MeetingChatMessage.meeting_id == meeting_id)
    )
    await session.execute(delete(Meeting).where(Meeting.id == meeting_id))
    await session.commit()

    await _cleanup_unreferenced_artifacts(session, settings, meeting_id, artifacts)
    logger.info("meeting %s deleted", meeting_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _cleanup_unreferenced_artifacts(
    session: AsyncSession,
    settings: Settings,
    meeting_id: str,
    artifacts: set[str],
) -> None:
    meetings_root = settings.meetings_dir.resolve()
    for relative_path in sorted(artifacts):
        still_referenced = (
            await session.execute(
                select(Meeting.id)
                .where(
                    or_(
                        Meeting.audio_mp3_path == relative_path,
                        Meeting.processing_wav_path == relative_path,
                    )
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        if still_referenced is not None:
            continue

        target = (meetings_root / relative_path).resolve()
        if meetings_root not in target.parents:
            logger.warning(
                "meeting %s: refusing to delete artifact outside the data directory", meeting_id
            )
            continue

        try:
            target.unlink(missing_ok=True)
            parent = target.parent
            if parent != meetings_root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            logger.warning(
                "meeting %s: artifact cleanup failed (database stays consistent)",
                meeting_id,
                exc_info=True,
            )


@router.get("/{meeting_id}/audio")
async def get_audio(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    """Stream the stored MP3. Only the persisted relative path is used; the WAV is
    never exposed and arbitrary filesystem paths can not be requested."""
    meeting = await _get_meeting(session, meeting_id)

    meetings_root = settings.meetings_dir.resolve()
    audio_path = (meetings_root / meeting.audio_mp3_path).resolve()
    if meetings_root not in audio_path.parents or not audio_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio not found")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename=f"{meeting_id}.mp3",
    )


@router.post(
    "/{meeting_id}/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def queue_analysis(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    refresh: Annotated[bool, Query(description="Rebuild with current aliases")] = False,
) -> AnalysisResponse:
    """Queue grounded analysis.

    Idempotent by default; a failed analysis may be retried. `refresh=true` rebuilds
    a finished analysis with the CURRENT speaker aliases (explicit user action, so
    no LLM quota is spent on every rename).
    """
    meeting = await _get_meeting(session, meeting_id)
    if meeting.status != MEETING_STATUS_COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Meeting transcript must be completed before analysis",
        )

    existing = (
        await session.execute(
            select(MeetingAnalysis).where(MeetingAnalysis.meeting_id == meeting_id)
        )
    ).scalar_one_or_none()

    if existing is not None and refresh:
        if existing.status in (ANALYSIS_STATUS_QUEUED, ANALYSIS_STATUS_PROCESSING):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Analysis is already running",
            )
        try:
            build_provider(settings)
        except LlmConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        existing.status = ANALYSIS_STATUS_QUEUED
        existing.analysis_error = None
        await session.commit()
        await session.refresh(existing)
        logger.info("analysis refresh queued for meeting %s", meeting_id)
        return await _analysis_response(session, existing)

    if existing is None:
        # Fail fast when no eligible provider is configured: the queue would only
        # produce a failed analysis otherwise.
        try:
            build_provider(settings)
        except LlmConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        analysis = MeetingAnalysis(meeting_id=meeting_id, status=ANALYSIS_STATUS_QUEUED)
        session.add(analysis)
        await session.commit()
        await session.refresh(analysis)
        logger.info("analysis queued for meeting %s", meeting_id)
        return await _analysis_response(session, analysis)

    if existing.status in (
        ANALYSIS_STATUS_QUEUED,
        ANALYSIS_STATUS_PROCESSING,
        ANALYSIS_STATUS_COMPLETED,
    ):
        return await _analysis_response(session, existing)

    # Explicit retry of a failed analysis.
    existing.status = ANALYSIS_STATUS_QUEUED
    existing.analysis_error = None
    await session.commit()
    await session.refresh(existing)
    logger.info("failed analysis requeued for meeting %s", meeting_id)
    return await _analysis_response(session, existing)


@router.get("/{meeting_id}/analysis", response_model=AnalysisResponse)
async def get_analysis(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AnalysisResponse:
    await _get_meeting(session, meeting_id)
    analysis = (
        await session.execute(
            select(MeetingAnalysis).where(MeetingAnalysis.meeting_id == meeting_id)
        )
    ).scalar_one_or_none()
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return await _analysis_response(session, analysis)


class ChatQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


class ChatMessageOut(BaseModel):
    role: str
    content: str
    # Safe provider metadata only (never keys or raw provider payloads).
    provider: str | None = None
    model: str | None = None
    created_at: str


class ChatConversationResponse(BaseModel):
    meeting_id: str
    messages: list[ChatMessageOut]


class ChatAnswerResponse(ChatConversationResponse):
    answer: str


def _chat_message_out(message: MeetingChatMessage) -> ChatMessageOut:
    return ChatMessageOut(
        role=message.role,
        content=message.content,
        provider=message.provider,
        model=message.model,
        created_at=_iso_utc(message.created_at),
    )


@router.get("/{meeting_id}/chat", response_model=ChatConversationResponse)
async def get_chat(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ChatConversationResponse:
    """Return the persisted Q&A conversation for a meeting (oldest first)."""
    await _get_meeting(session, meeting_id)
    messages = await list_chat_messages(session, meeting_id)
    return ChatConversationResponse(
        meeting_id=meeting_id,
        messages=[_chat_message_out(message) for message in messages],
    )


@router.post("/{meeting_id}/chat", response_model=ChatAnswerResponse)
async def chat_about_meeting(
    request: ChatQuestionRequest,
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatAnswerResponse:
    """Answer a question about a completed meeting, grounded in its transcript.

    Synchronous (no worker queue): the answer is produced with the configured
    OpenAI-compatible provider and the exchange is persisted so the conversation
    survives a refresh. No secret is ever returned.
    """
    meeting = await _get_meeting(session, meeting_id)
    if meeting.status != MEETING_STATUS_COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Meeting transcript must be completed before asking questions",
        )

    try:
        answer = await answer_meeting_question(
            session, meeting_id, request.question, settings
        )
    except LlmConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LlmProviderError as exc:
        logger.warning("meeting chat failed for %s: %s", meeting_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Meeting assistant is temporarily unavailable.",
        ) from exc

    messages = await list_chat_messages(session, meeting_id)
    return ChatAnswerResponse(
        meeting_id=meeting_id,
        answer=answer.answer,
        messages=[_chat_message_out(message) for message in messages],
    )


@router.delete("/{meeting_id}/chat", status_code=status.HTTP_204_NO_CONTENT)
async def clear_chat(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    """Clear the persisted Q&A conversation for a meeting."""
    await _get_meeting(session, meeting_id)
    await clear_chat_messages(session, meeting_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{meeting_id}", response_model=MeetingStatus)
async def get_meeting(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> MeetingStatus:
    meeting = await _get_meeting(session, meeting_id)
    return _meeting_status(meeting, await _turn_count(session, meeting_id) > 0)


@router.get("/{meeting_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(
    meeting_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TranscriptResponse:
    meeting = await _get_meeting(session, meeting_id)
    if meeting.status != MEETING_STATUS_COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Transcript not available (status: {meeting.status})",
        )

    turns = (
        (
            await session.execute(
                select(TranscriptTurn)
                .where(TranscriptTurn.meeting_id == meeting_id)
                .order_by(TranscriptTurn.ordinal)
            )
        )
        .scalars()
        .all()
    )

    speakers = sorted({turn.speaker for turn in turns if turn.speaker != UNKNOWN_SPEAKER})
    return TranscriptResponse(
        meeting_id=meeting.id,
        status=meeting.status,
        duration_seconds=meeting.duration_seconds,
        speakers=speakers,
        unresolved_label=UNKNOWN_SPEAKER,
        unresolved_turns=sum(1 for turn in turns if turn.speaker == UNKNOWN_SPEAKER),
        turns=[
            TranscriptTurnOut(
                ordinal=turn.ordinal,
                speaker=turn.speaker,
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                text=turn.text,
            )
            for turn in turns
        ],
    )


async def _analysis_response(session: AsyncSession, analysis: MeetingAnalysis) -> AnalysisResponse:
    """Serialize the analysis; timestamps come from persisted transcript rows."""
    response = AnalysisResponse(
        meeting_id=analysis.meeting_id,
        status=analysis.status,
        provider=analysis.provider,
        model=analysis.model,
        analysis_error=analysis.analysis_error,
        input_chars=analysis.input_chars,
        latency_seconds=analysis.latency_seconds,
        repair_attempts=analysis.repair_attempts,
    )
    if analysis.status != ANALYSIS_STATUS_COMPLETED:
        return response

    payload = payload_from_row(analysis)
    if payload is None:
        return response

    rows = (
        await session.execute(
            select(TranscriptTurn.ordinal, TranscriptTurn.start_seconds).where(
                TranscriptTurn.meeting_id == analysis.meeting_id
            )
        )
    ).all()
    start_seconds = {ordinal: seconds for ordinal, seconds in rows}

    def first_timestamp(ordinals: list[int]) -> float:
        return round(start_seconds[ordinals[0]], 3)

    return response.model_copy(
        update={
            "summary": payload.summary,
            "key_points": [
                KeyPointOut(
                    text=key_point.text,
                    source_turn_ordinals=key_point.source_turn_ordinals,
                    timestamp_seconds=first_timestamp(key_point.source_turn_ordinals),
                )
                for key_point in payload.key_points
            ],
            "topics": payload.topics,
            "decisions": [
                DecisionOut(
                    text=decision.text,
                    source_turn_ordinals=decision.source_turn_ordinals,
                    timestamp_seconds=first_timestamp(decision.source_turn_ordinals),
                )
                for decision in payload.decisions
            ],
            "action_items": [
                ActionItemOut(
                    task=item.task,
                    owner=item.owner,
                    due_date_text=item.due_date_text,
                    source_turn_ordinals=item.source_turn_ordinals,
                    timestamp_seconds=first_timestamp(item.source_turn_ordinals),
                )
                for item in payload.action_items
            ],
            "important_moments": [
                ImportantMomentOut(
                    title=moment.title,
                    description=moment.description,
                    source_turn_ordinal=moment.source_turn_ordinal,
                    timestamp_seconds=round(
                        start_seconds[moment.source_turn_ordinal], 3
                    ),
                )
                for moment in payload.important_moments
            ],
        }
    )
