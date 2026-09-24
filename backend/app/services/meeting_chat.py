"""Synchronous, grounded Q&A over a completed meeting.

Unlike transcription and analysis this is not queued through the worker: the answer
is produced in the API process with the same LLM provider, so the UI can stream a
reply immediately. Only the transcript and the grounded analysis summary are sent.

The conversation is persisted (``meeting_chat_messages``) so it survives a page
refresh; only text and safe provider metadata are stored.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.models import (
    ANALYSIS_STATUS_COMPLETED,
    CHAT_ROLE_ASSISTANT,
    CHAT_ROLE_USER,
    MEETING_STATUS_COMPLETED,
    Meeting,
    MeetingAnalysis,
    MeetingChatMessage,
    TranscriptTurn,
)
from app.services.analysis_pipeline import extract_json_object
from app.services.analysis_schema import TranscriptTurnView
from app.services.chat_prompt import CHAT_SYSTEM_PROMPT, build_chat_user_prompt
from app.services.llm_provider import LlmProviderError, build_provider
from app.services.llm_runtime import run_llm
from app.services.speaker_aliases import list_aliases

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12
MAX_HISTORY_CHARS = 4_000


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    provider: str
    model: str
    sources: list[int] = field(default_factory=list)


class MeetingNotReadyError(RuntimeError):
    """The meeting has no completed transcript yet."""


async def list_chat_messages(
    session: AsyncSession, meeting_id: str
) -> list[MeetingChatMessage]:
    rows = (
        (
            await session.execute(
                select(MeetingChatMessage)
                .where(MeetingChatMessage.meeting_id == meeting_id)
                .order_by(MeetingChatMessage.id)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


async def clear_chat_messages(session: AsyncSession, meeting_id: str) -> int:
    result = await session.execute(
        delete(MeetingChatMessage).where(MeetingChatMessage.meeting_id == meeting_id)
    )
    await session.commit()
    return result.rowcount or 0


async def answer_meeting_question(
    session: AsyncSession,
    meeting_id: str,
    question: str,
    settings: Settings,
) -> ChatAnswer:
    meeting = await session.get(Meeting, meeting_id)
    if meeting is None or meeting.status != MEETING_STATUS_COMPLETED:
        raise MeetingNotReadyError("meeting transcript is not completed")

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

    analysis = (
        await session.execute(
            select(MeetingAnalysis).where(MeetingAnalysis.meeting_id == meeting_id)
        )
    ).scalar_one_or_none()
    summary = (
        analysis.summary
        if analysis is not None and analysis.status == ANALYSIS_STATUS_COMPLETED
        else None
    )

    stored = await list_chat_messages(session, meeting_id)
    history = [(message.role, message.content) for message in stored]

    user_prompt = build_chat_user_prompt(turns, summary, _trim_history(history), question)
    if len(user_prompt) > settings.llm_max_transcript_chars:
        raise LlmProviderError("meeting is too large for a single chat request")

    provider = build_provider(settings)
    # Bounded, dedicated pool: chat traffic never starves uploads or the event loop.
    response = await run_llm(
        provider,
        system_prompt=CHAT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        session_id=meeting_id,
        settings=settings,
    )
    valid_ordinals = {row.ordinal for row in rows}
    answer_text, sources = _parse_answer(
        response.content, valid_ordinals=valid_ordinals
    )

    # Persist the exchange only once the answer exists: the stored history is always
    # complete question/answer pairs, so a failed call leaves nothing behind.
    session.add(
        MeetingChatMessage(meeting_id=meeting_id, role=CHAT_ROLE_USER, content=question)
    )
    session.add(
        MeetingChatMessage(
            meeting_id=meeting_id,
            role=CHAT_ROLE_ASSISTANT,
            content=answer_text,
            provider=response.provider,
            model=response.model,
            sources_json=json.dumps(sources),
        )
    )
    await session.commit()

    return ChatAnswer(
        answer=answer_text,
        provider=response.provider,
        model=response.model,
        sources=sources,
    )


def _parse_answer(content: str, *, valid_ordinals: set[int]) -> tuple[str, list[int]]:
    """Extract the Turkish answer and its evidence ordinals from the model output.

    The model is asked for JSON, but a provider that ignores that is handled
    gracefully: the raw text becomes the answer with no sources (never a crash and
    never an invented timestamp).
    """
    fallback = content.strip()
    try:
        data = json.loads(extract_json_object(content))
    except ValueError:
        return fallback, []
    if not isinstance(data, dict):
        return fallback, []

    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return fallback, []

    sources: list[int] = []
    raw_sources = data.get("source_turn_ordinals")
    if isinstance(raw_sources, list):
        for item in raw_sources:
            # bool is an int subclass: never accept True/False as an ordinal.
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            if item in valid_ordinals and item not in sources:
                sources.append(item)
    return answer.strip(), sources


def _trim_history(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Keep only the most recent exchanges, bounded by count and total characters."""
    kept: list[tuple[str, str]] = []
    total = 0
    for role, content in reversed(history[-MAX_HISTORY_MESSAGES:]):
        total += len(content)
        if kept and total > MAX_HISTORY_CHARS:
            break
        kept.append((role, content))
    kept.reverse()
    return kept
