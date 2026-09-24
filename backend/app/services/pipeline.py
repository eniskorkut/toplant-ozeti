"""Meeting processing pipeline: claim -> STT -> diarization -> merge -> persist.

Runs only inside the worker process (never inside an HTTP request). The claim is an
atomic state transition (`queued -> processing`) so a job can not be processed twice.
On failure the meeting is marked `failed` with a short message and any partial
transcript rows are removed, so a failed job never looks completed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    Meeting,
    MeetingLiveSpeaker,
    MeetingSpeakerAlias,
    TranscriptTurn,
)
from app.services.speaker_aliases import list_aliases
from app.services.speaker_matching import reconcile_final_aliases, remap_final_speakers
from app.services.transcription import ProviderError, build_provider, form_turns

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500


async def requeue_stale_meetings(session: AsyncSession, lease_seconds: float = 0.0) -> int:
    """Requeue meetings stuck in `processing` beyond the lease.

    Lease-based rather than "everything on startup", so it is safe with multiple
    worker replicas: a meeting another worker is actively processing is never stolen.
    Rows with no lease timestamp (legacy/unknown) count as stale.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=max(0.0, lease_seconds))
    result = await session.execute(
        update(Meeting)
        .where(
            Meeting.status == MEETING_STATUS_PROCESSING,
            or_(
                Meeting.processing_started_at.is_(None),
                Meeting.processing_started_at < cutoff,
            ),
        )
        .values(status=MEETING_STATUS_QUEUED, processing_started_at=None)
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
        .values(
            status=MEETING_STATUS_PROCESSING,
            processing_error=None,
            processing_started_at=datetime.now(UTC),
        )
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
    requested_provider = meeting.requested_transcription_provider

    try:
        audio_path = settings.meetings_dir / processing_wav_path
        if not audio_path.exists():
            raise FileNotFoundError(f"processing audio missing: {audio_path}")

        live_timelines = await _load_live_timelines(session, meeting_id)
        live_aliases = (
            await list_aliases(session, meeting_id) if live_timelines else None
        )

        output = await asyncio.to_thread(
            _run_inference,
            audio_path,
            requested_speaker_count,
            settings,
            requested_provider,
            live_timelines,
            live_aliases,
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
        if output.get("final_aliases") is not None:
            await session.execute(
                delete(MeetingSpeakerAlias).where(MeetingSpeakerAlias.meeting_id == meeting_id)
            )
            for canonical, display in output["final_aliases"].items():
                session.add(
                    MeetingSpeakerAlias(
                        meeting_id=meeting_id,
                        canonical_speaker=canonical,
                        display_name=display,
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


async def _load_live_timelines(
    session: AsyncSession, meeting_id: str
) -> dict[str, list[tuple[float, float]]]:
    """Canonical live speaker timelines persisted at upload time, if any."""
    rows = (
        await session.execute(
            select(MeetingLiveSpeaker).where(MeetingLiveSpeaker.meeting_id == meeting_id)
        )
    ).scalars()
    timelines: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            raw = json.loads(row.intervals_json)
            timelines[row.canonical_speaker] = [
                (float(start), float(end)) for start, end in raw
            ]
        except (TypeError, ValueError):
            continue
    return timelines


def _run_inference(
    audio_path,
    requested_speaker_count: int | None,
    settings: Settings,
    requested_provider: str | None = None,
    live_timelines: dict[str, list[tuple[float, float]]] | None = None,
    live_aliases: dict[str, str] | None = None,
) -> dict:
    """Blocking inference step (runs in a worker thread).

    The provider produces normalized words (anonymous speaker ids, possibly absent);
    turn formation is shared by every provider and mirrors the local merge semantics.
    For live-recorded meetings the final provider speakers are reconciled onto the
    canonical live labels so stable Kişi numbers and aliases survive the final pass.
    """
    import wave

    with wave.open(str(audio_path), "rb") as wav_file:
        duration_seconds = wav_file.getnframes() / wav_file.getframerate()

    effective_settings = settings
    if requested_provider and requested_provider != settings.transcription_provider:
        # Per-meeting override; null keeps the configured server default.
        effective_settings = settings.model_copy(
            update={"transcription_provider": requested_provider}
        )
    provider = build_provider(effective_settings)
    result = provider.transcribe(
        audio_path,
        requested_speaker_count=requested_speaker_count,
        settings=effective_settings,
    )

    speaker_labels: dict[str, str] | None = None
    final_aliases: dict[str, str] | None = None
    if live_timelines and result.provider == "elevenlabs":
        provider_intervals: dict[str, list[tuple[float, float]]] = {}
        for word in result.words:
            if word.speaker_id is None:
                continue
            provider_intervals.setdefault(word.speaker_id, []).append((word.start, word.end))
        if provider_intervals:
            speaker_labels = remap_final_speakers(live_timelines, provider_intervals)
            if live_aliases is not None:
                final_aliases = reconcile_final_aliases(
                    live_timelines, provider_intervals, live_aliases
                )

    turns = form_turns(result.words, speaker_labels=speaker_labels)

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
        "final_aliases": final_aliases,
    }


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (ProviderError, FileNotFoundError)):
        message = str(exc)
    else:
        message = f"{type(exc).__name__}: {exc}"
    return message[:MAX_ERROR_LENGTH]
