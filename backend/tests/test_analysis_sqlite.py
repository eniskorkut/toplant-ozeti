"""SQLite hardening and stale-job recovery tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db import Database
from app.models import (
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    Meeting,
    MeetingAnalysis,
)
from app.services.analysis_pipeline import requeue_stale_analyses
from app.services.pipeline import requeue_stale_meetings


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'pragma.sqlite'}")
    asyncio.run(db.init())
    return db


def test_sqlite_pragmas_are_applied(database: Database) -> None:
    async def run() -> tuple[str, int, int]:
        async with database.session_factory() as session:
            from sqlalchemy import text

            journal = (await session.execute(text("PRAGMA journal_mode"))).scalar_one()
            busy = (await session.execute(text("PRAGMA busy_timeout"))).scalar_one()
            foreign_keys = (await session.execute(text("PRAGMA foreign_keys"))).scalar_one()
            return str(journal).lower(), int(busy), int(foreign_keys)

    journal, busy_timeout, foreign_keys = asyncio.run(run())
    assert journal == "wal"
    assert busy_timeout == 5000
    assert foreign_keys == 1


def test_foreign_keys_are_enforced(database: Database) -> None:
    async def run() -> None:
        from sqlalchemy.exc import IntegrityError

        from app.models import TranscriptTurn

        async with database.session_factory() as session:
            session.add(
                TranscriptTurn(
                    meeting_id="does-not-exist",
                    ordinal=0,
                    speaker="Kişi 1",
                    start_seconds=0.0,
                    end_seconds=1.0,
                    text="x",
                )
            )
            with pytest.raises(IntegrityError):
                await session.commit()

    asyncio.run(run())


def add_meeting(database: Database, meeting_id: str, status: str) -> None:
    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status=status,
                    audio_mp3_path=f"{meeting_id}/meeting.mp3",
                    processing_wav_path=f"{meeting_id}/processing.wav",
                )
            )
            await session.commit()

    asyncio.run(run())


def test_stale_transcription_processing_is_requeued(database: Database) -> None:
    add_meeting(database, "stale", MEETING_STATUS_PROCESSING)
    add_meeting(database, "queued", MEETING_STATUS_QUEUED)

    async def run() -> tuple[int, dict[str, str]]:
        async with database.session_factory() as session:
            count = await requeue_stale_meetings(session)
            statuses = {
                meeting.id: meeting.status
                for meeting in (await session.execute(select(Meeting))).scalars()
            }
            return count, statuses

    count, statuses = asyncio.run(run())
    assert count == 1
    assert statuses["stale"] == MEETING_STATUS_QUEUED
    assert statuses["queued"] == MEETING_STATUS_QUEUED


def test_stale_analysis_processing_is_requeued(database: Database) -> None:
    add_meeting(database, "m-analysis", MEETING_STATUS_QUEUED)

    async def run() -> tuple[int, str]:
        async with database.session_factory() as session:
            session.add(
                MeetingAnalysis(meeting_id="m-analysis", status=ANALYSIS_STATUS_PROCESSING)
            )
            await session.commit()
            count = await requeue_stale_analyses(session)
            analysis = (
                await session.execute(
                    select(MeetingAnalysis).where(MeetingAnalysis.meeting_id == "m-analysis")
                )
            ).scalar_one()
            return count, analysis.status

    count, status = asyncio.run(run())
    assert count == 1
    assert status == ANALYSIS_STATUS_QUEUED
