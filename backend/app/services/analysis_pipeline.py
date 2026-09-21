"""Grounded analysis pipeline: claim -> provider -> validate -> repair -> persist.

LLM failures never touch the transcript: this module only writes the MeetingAnalysis
row and (on failure) removes nothing from the meeting/transcript tables.
"""

from __future__ import annotations

import json
import logging
import re

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_COMPLETED,
    Meeting,
    MeetingAnalysis,
    TranscriptTurn,
)
from app.services.analysis_prompt import (
    SYSTEM_PROMPT,
    build_repair_prompt,
    build_user_prompt,
)
from app.services.analysis_schema import (
    AnalysisPayload,
    TranscriptTurnView,
    validate_payload,
)
from app.services.llm_provider import (
    LlmConfigurationError,
    LlmProviderError,
    build_provider,
)
from app.services.speaker_aliases import list_aliases

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(content: str) -> str:
    """Best-effort extraction of the JSON object from assistant content."""
    fenced = _FENCE_RE.search(content)
    candidate = fenced.group(1) if fenced else content
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return candidate.strip()
    return candidate[start : end + 1]


async def requeue_stale_analyses(session: AsyncSession) -> int:
    """Recovery for the single-worker MVP: processing -> queued on startup."""
    result = await session.execute(
        update(MeetingAnalysis)
        .where(MeetingAnalysis.status == ANALYSIS_STATUS_PROCESSING)
        .values(status=ANALYSIS_STATUS_QUEUED)
    )
    await session.commit()
    return result.rowcount or 0


async def claim_next_analysis(session: AsyncSession) -> MeetingAnalysis | None:
    candidate_id = (
        await session.execute(
            select(MeetingAnalysis.id)
            .where(MeetingAnalysis.status == ANALYSIS_STATUS_QUEUED)
            .order_by(MeetingAnalysis.created_at)
            .limit(1)
        )
    ).scalar_one_or_none()
    if candidate_id is None:
        return None

    claimed = await session.execute(
        update(MeetingAnalysis)
        .where(MeetingAnalysis.id == candidate_id, MeetingAnalysis.status == ANALYSIS_STATUS_QUEUED)
        .values(status=ANALYSIS_STATUS_PROCESSING, analysis_error=None)
    )
    await session.commit()
    if claimed.rowcount != 1:
        return None
    return await session.get(MeetingAnalysis, candidate_id)


async def process_analysis(
    session: AsyncSession, analysis: MeetingAnalysis, settings: Settings
) -> None:
    analysis_id = analysis.id
    meeting_id = analysis.meeting_id
    repair_attempts = 0

    try:
        meeting = await session.get(Meeting, meeting_id)
        if meeting is None or meeting.status != MEETING_STATUS_COMPLETED:
            raise LlmConfigurationError("meeting transcript is not completed")

        rows = (
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
        # Meeting-scoped aliases: human-readable prose uses the user's display
        # names, while canonical labels stay the grounding identity.
        aliases = await list_aliases(session, meeting_id)
        turns = [
            TranscriptTurnView(
                ordinal=row.ordinal,
                speaker=row.speaker,
                start_seconds=row.start_seconds,
                text=row.text,
                alias=aliases.get(row.speaker),
            )
            for row in rows
        ]
        user_prompt = build_user_prompt(turns)
        if len(user_prompt) > settings.llm_max_transcript_chars:
            raise LlmProviderError("transcript too large for single-pass analysis")

        provider = build_provider(settings)
        response = provider.analyze(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            session_id=meeting_id,
        )

        raw = extract_json_object(response.content)
        payload, errors = validate_payload(raw, turns)

        if payload is None:
            # Exactly one repair attempt with the validation errors and prior output.
            repair_attempts = 1
            repaired = provider.analyze(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_repair_prompt(turns, errors),
                session_id=meeting_id,
            )
            payload, errors = validate_payload(extract_json_object(repaired.content), turns)
            if payload is None:
                raise LlmProviderError(
                    "analysis output failed validation after one repair attempt: "
                    + "; ".join(errors[:5])
                )

        stored = await session.get(MeetingAnalysis, analysis_id)
        if stored is None:
            raise LlmConfigurationError("analysis row disappeared")
        stored.status = ANALYSIS_STATUS_COMPLETED
        stored.provider = response.provider
        stored.model = response.model
        stored.summary = payload.summary
        stored.key_points_json = json.dumps(
            [item.model_dump() for item in payload.key_points], ensure_ascii=False
        )
        stored.topics_json = json.dumps(payload.topics, ensure_ascii=False)
        stored.decisions_json = json.dumps(
            [item.model_dump() for item in payload.decisions], ensure_ascii=False
        )
        stored.action_items_json = json.dumps(
            [item.model_dump() for item in payload.action_items], ensure_ascii=False
        )
        stored.important_moments_json = json.dumps(
            [item.model_dump() for item in payload.important_moments], ensure_ascii=False
        )
        stored.input_chars = len(user_prompt)
        stored.latency_seconds = response.latency_seconds
        stored.repair_attempts = repair_attempts
        stored.analysis_error = None
        await session.commit()
        logger.info(
            "analysis %s completed for meeting %s (provider=%s, chars=%d, repair_attempts=%d)",
            analysis_id,
            meeting_id,
            response.provider,
            len(user_prompt),
            repair_attempts,
        )
    except Exception as exc:
        await session.rollback()
        failed = await session.get(MeetingAnalysis, analysis_id)
        if failed is not None:
            failed.status = ANALYSIS_STATUS_FAILED
            failed.analysis_error = _safe_error_message(exc)
            failed.repair_attempts = repair_attempts
            await session.commit()
        logger.warning("analysis %s failed: %s", analysis_id, _safe_error_message(exc))


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (LlmProviderError, ValueError)):
        message = str(exc)
    else:
        message = f"{type(exc).__name__}: {exc}"
    return message[:MAX_ERROR_LENGTH]


def payload_from_row(row: MeetingAnalysis) -> AnalysisPayload | None:
    if row.summary is None:
        return None
    return AnalysisPayload(
        summary=row.summary,
        key_points=json.loads(row.key_points_json or "[]"),
        topics=json.loads(row.topics_json or "[]"),
        decisions=json.loads(row.decisions_json or "[]"),
        action_items=json.loads(row.action_items_json or "[]"),
        important_moments=json.loads(row.important_moments_json or "[]"),
    )
