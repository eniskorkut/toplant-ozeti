"""Meeting processing pipeline: claim -> STT -> diarization -> merge -> persist.

Runs only inside the worker process (never inside an HTTP request). The claim is an
atomic state transition (`queued -> processing`) so a job can not be processed twice.
On failure the meeting is marked `failed` with a short message and any partial
transcript rows are removed, so a failed job never looks completed.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    Meeting,
    TranscriptTurn,
)
from app.services.transcription import ProviderError, build_provider, form_turns

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500


async def requeue_stale_meetings(session: AsyncSession) -> int:
    """Recovery for the single-worker MVP: processing -> queued on worker startup.

    Assumes exactly ONE worker instance. With multiple workers this would requeue a
    job that another worker is still processing.
    """
    result = await session.execute(
        update(Meeting)
        .where(Meeting.status == MEETING_STATUS_PROCESSING)
        .values(status=MEETING_STATUS_QUEUED)
    )
    await session.commit()
    return result.rowcount or 0


async def claim_next_meeting(session: AsyncSession) -> Meeting | None:
    """Atomically claim the oldest queued meeting, or return None."""
    candidate_id = (
        await session.execute(
            select(Meeting.id)
            .where(Meeting.status == MEETING_STATUS_QUEUED)
            .order_by(Meeting.created_at)
            .limit(1)
        )
    ).scalar_one_or_none()
    if candidate_id is None:
        return None

    claimed = await session.execute(
        update(Meeting)
        .where(Meeting.id == candidate_id, Meeting.status == MEETING_STATUS_QUEUED)
        .values(status=MEETING_STATUS_PROCESSING, processing_error=None)
    )
    await session.commit()
    if claimed.rowcount != 1:
        return None
    return await session.get(Meeting, candidate_id)


async def process_meeting(session: AsyncSession, meeting: Meeting, settings: Settings) -> None:
    """Run the full pipeline for a claimed meeting and persist the result.

    CPU-heavy inference runs in a worker thread so the worker event loop stays
    responsive while inference is running.
    """
    # Snapshot plain values before any await: rollback expires ORM instances and
    # touching them afterwards would require a lazy load outside greenlet context.
    meeting_id = meeting.id
    processing_wav_path = meeting.processing_wav_path
    requested_speaker_count = meeting.requested_speaker_count

    try:
        audio_path = settings.meetings_dir / processing_wav_path
        if not audio_path.exists():
            raise FileNotFoundError(f"processing audio missing: {audio_path}")

        output = await asyncio.to_thread(
            _run_inference, audio_path, requested_speaker_count, settings
        )

        await session.execute(delete(TranscriptTurn).where(TranscriptTurn.meeting_id == meeting_id))
        for ordinal, turn in enumerate(output["turns"]):
            session.add(
                TranscriptTurn(
                    meeting_id=meeting_id,
                    ordinal=ordinal,
                    speaker=turn["speaker"],
                    start_seconds=turn["start"],
                    end_seconds=turn["end"],
                    text=turn["text"],
                )
            )
        meeting.status = MEETING_STATUS_COMPLETED
        meeting.processing_error = None
        meeting.duration_seconds = output["duration_seconds"]
        meeting.transcription_provider = output["provider"]
        meeting.transcription_model = output["model"]
        await session.commit()
        logger.info(
            "meeting %s completed: %d turns, %d unresolved words, %.1fs provider=%s",
            meeting_id,
            len(output["turns"]),
            output["unresolved_words"],
            output["inference_seconds"],
            output["provider"],
        )
    except Exception as exc:
        await session.rollback()
        await session.execute(delete(TranscriptTurn).where(TranscriptTurn.meeting_id == meeting_id))
        failed = await session.get(Meeting, meeting_id)
        if failed is not None:
            failed.status = MEETING_STATUS_FAILED
            failed.processing_error = _safe_error_message(exc)
        await session.commit()
        logger.warning("meeting %s failed: %s", meeting_id, _safe_error_message(exc))


def _run_inference(
    audio_path, requested_speaker_count: int | None, settings: Settings
) -> dict:
    """Blocking inference step (runs in a worker thread).

    The provider produces normalized words (anonymous speaker ids, possibly absent);
    turn formation is shared by every provider and mirrors the local merge semantics.
    """
    import wave

    with wave.open(str(audio_path), "rb") as wav_file:
        duration_seconds = wav_file.getnframes() / wav_file.getframerate()

    provider = build_provider(settings)
    result = provider.transcribe(
        audio_path,
        requested_speaker_count=requested_speaker_count,
        settings=settings,
    )
    turns = form_turns(result.words)

    unresolved = sum(1 for word in result.words if word.speaker_id is None)
    return {
        "duration_seconds": round(duration_seconds, 3),
        "turns": [
            {
                "speaker": turn.speaker,
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "text": turn.text,
            }
            for turn in turns
        ],
        "unresolved_words": unresolved,
        "assigned_words": len(result.words) - unresolved,
        "speaker_count": len({word.speaker_id for word in result.words if word.speaker_id}),
        "inference_seconds": result.latency_seconds,
        "language": result.language,
        "provider": result.provider,
        "model": result.model,
    }


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (ProviderError, FileNotFoundError)):
        message = str(exc)
    else:
        message = f"{type(exc).__name__}: {exc}"
    return message[:MAX_ERROR_LENGTH]
