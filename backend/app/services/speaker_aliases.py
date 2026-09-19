"""Meeting-scoped speaker display aliases.

An alias is a display name for one canonical speaker label inside ONE meeting
("Kişi 1" -> "Ahmet"). It is not an identity: no global person, no embedding, no
cross-meeting lookup. Validation is shared by the live-session and persisted
meeting endpoints so both behave identically.
"""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import MeetingSpeakerAlias, TranscriptTurn

MAX_DISPLAY_NAME_LENGTH = 50
CANONICAL_SPEAKER_PATTERN = re.compile(r"^Kişi \d{1,2}$")


class AliasValidationError(ValueError):
    """Raised when a display name or canonical speaker label is invalid."""


def validate_display_name(raw: str) -> str:
    """Trimmed, non-empty, length-bounded and free of control characters."""
    name = (raw or "").strip()
    if not name:
        raise AliasValidationError("display_name must not be empty")
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise AliasValidationError(
            f"display_name must be at most {MAX_DISPLAY_NAME_LENGTH} characters"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise AliasValidationError("display_name must not contain control characters")
    return name


def validate_canonical_speaker(raw: str) -> str:
    label = (raw or "").strip()
    if not CANONICAL_SPEAKER_PATTERN.match(label):
        raise AliasValidationError("canonical speaker must look like 'Kişi 1'")
    return label


async def list_aliases(session: AsyncSession, meeting_id: str) -> dict[str, str]:
    rows = (
        await session.execute(
            select(MeetingSpeakerAlias).where(MeetingSpeakerAlias.meeting_id == meeting_id)
        )
    ).scalars()
    return {row.canonical_speaker: row.display_name for row in rows}


async def set_alias(
    session: AsyncSession,
    *,
    meeting_id: str,
    canonical_speaker: str,
    display_name: str,
) -> MeetingSpeakerAlias:
    existing = (
        await session.execute(
            select(MeetingSpeakerAlias).where(
                MeetingSpeakerAlias.meeting_id == meeting_id,
                MeetingSpeakerAlias.canonical_speaker == canonical_speaker,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        existing = MeetingSpeakerAlias(
            meeting_id=meeting_id,
            canonical_speaker=canonical_speaker,
            display_name=display_name,
        )
        session.add(existing)
    else:
        existing.display_name = display_name
    await session.commit()
    await session.refresh(existing)
    return existing


async def clear_alias(session: AsyncSession, *, meeting_id: str, canonical_speaker: str) -> bool:
    existing = (
        await session.execute(
            select(MeetingSpeakerAlias).where(
                MeetingSpeakerAlias.meeting_id == meeting_id,
                MeetingSpeakerAlias.canonical_speaker == canonical_speaker,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        return False
    await session.delete(existing)
    await session.commit()
    return True


async def migrate_aliases(
    session: AsyncSession,
    *,
    meeting_id: str,
    aliases: dict[str, str],
) -> dict[str, str]:
    """Copy live-session aliases onto the freshly created meeting row."""
    migrated: dict[str, str] = {}
    for canonical_speaker, display_name in aliases.items():
        try:
            label = validate_canonical_speaker(canonical_speaker)
            name = validate_display_name(display_name)
        except AliasValidationError:
            continue
        await set_alias(
            session,
            meeting_id=meeting_id,
            canonical_speaker=label,
            display_name=name,
        )
        migrated[label] = name
    return migrated


async def speaker_labels_in_transcript(session: AsyncSession, meeting_id: str) -> set[str]:
    rows = (
        await session.execute(
            select(TranscriptTurn.speaker).where(TranscriptTurn.meeting_id == meeting_id).distinct()
        )
    ).scalars()
    return set(rows)
