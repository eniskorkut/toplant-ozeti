"""Meeting processing API: queue processing, poll status, fetch the transcript."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi import Path as PathParam
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import exists, select
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
    TranscriptTurn,
)
from app.services.analysis_pipeline import payload_from_row
from app.services.llm_provider import LlmConfigurationError, build_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/meetings", tags=["meetings"])

MAX_SPEAKER_COUNT = 12


class ProcessRequest(BaseModel):
    speaker_count: int | None = Field(
        default=None,
        ge=1,
        le=MAX_SPEAKER_COUNT,
        description="Optional known speaker count; null selects automatic clustering.",
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


class MeetingList(BaseModel):
    meetings: list[MeetingSummary]
    count: int


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
        created_at=meeting.created_at.isoformat(),
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

    meeting.status = MEETING_STATUS_QUEUED
    meeting.requested_speaker_count = request.speaker_count
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
            created_at=meeting.created_at.isoformat(),
            duration_seconds=meeting.duration_seconds,
            status=meeting.status,
            requested_speaker_count=meeting.requested_speaker_count,
            has_transcript=bool(turns),
            analysis_status=analysis_status,
        )
        for meeting, turns, analysis_status in rows
    ]
    return MeetingList(meetings=meetings, count=len(meetings))


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
) -> AnalysisResponse:
    """Queue grounded analysis. Idempotent; a failed analysis may be retried."""
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
