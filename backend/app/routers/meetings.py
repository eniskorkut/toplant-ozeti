"""Meeting processing API: queue processing, poll status, fetch the transcript."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Path as PathParam
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    UNKNOWN_SPEAKER,
    Meeting,
    TranscriptTurn,
)

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
