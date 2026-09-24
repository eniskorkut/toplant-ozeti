"""Lease-based stale recovery and worker concurrency."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select

from app.config import Settings
from app.db import Database
from app.models import (
    ANALYSIS_STATUS_PROCESSING,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    Meeting,
    MeetingAnalysis,
)
from app.services.analysis_pipeline import requeue_stale_analyses
from app.services.pipeline import claim_next_meeting, requeue_stale_meetings
from app.worker import worker_loop


def add_meeting(
    database: Database,
    meeting_id: str,
    status: str,
    *,
    processing_started_at: datetime | None = None,
) -> None:
    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status=status,
                    audio_mp3_path=f"{meeting_id}/meeting.mp3",
                    processing_wav_path=f"{meeting_id}/processing.wav",
                    processing_started_at=processing_started_at,
                )
            )
            await session.commit()

    asyncio.run(run())


def test_fresh_processing_is_not_requeued(database: Database) -> None:
    add_meeting(
        database, "fresh", MEETING_STATUS_PROCESSING, processing_started_at=datetime.now(UTC)
    )

    async def run() -> int:
        async with database.session_factory() as session:
            return await requeue_stale_meetings(session, lease_seconds=3600)

    # Another worker is still processing it: the lease protects it.
    assert asyncio.run(run()) == 0


def test_expired_lease_is_requeued(database: Database) -> None:
    add_meeting(
        database,
        "stale",
        MEETING_STATUS_PROCESSING,
        processing_started_at=datetime.now(UTC) - timedelta(hours=2),
    )

    async def run() -> tuple[int, str]:
        async with database.session_factory() as session:
            count = await requeue_stale_meetings(session, lease_seconds=3600)
            meeting = (
                await session.execute(select(Meeting).where(Meeting.id == "stale"))
            ).scalar_one()
            return count, meeting.status

    count, status = asyncio.run(run())
    assert count == 1
    assert status == MEETING_STATUS_QUEUED


def test_claim_sets_processing_lease(database: Database) -> None:
    add_meeting(database, "claim", MEETING_STATUS_QUEUED)

    async def run() -> datetime | None:
        async with database.session_factory() as session:
            meeting = await claim_next_meeting(session)
            assert meeting is not None
            return meeting.processing_started_at

    assert asyncio.run(run()) is not None


def test_analysis_lease_protects_active_jobs(database: Database) -> None:
    add_meeting(database, "m", MEETING_STATUS_QUEUED)

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                MeetingAnalysis(
                    meeting_id="m",
                    status=ANALYSIS_STATUS_PROCESSING,
                    processing_started_at=datetime.now(UTC),
                )
            )
            await session.commit()

    asyncio.run(seed())

    async def run() -> int:
        async with database.session_factory() as session:
            return await requeue_stale_analyses(session, lease_seconds=3600)

    assert asyncio.run(run()) == 0


def test_worker_concurrency_runs_jobs_in_parallel(
    monkeypatch, tmp_path: Path
) -> None:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'concurrency.db'}"
    settings = Settings(
        data_dir=tmp_path / "meetings",
        database_url=database_url,
        worker_concurrency=2,
        worker_poll_seconds=0.01,
    )
    seed_db = Database(database_url)
    asyncio.run(seed_db.init())

    async def seed() -> None:
        async with seed_db.session_factory() as session:
            for index in range(2):
                session.add(
                    Meeting(
                        id=f"c{index}",
                        status=MEETING_STATUS_QUEUED,
                        audio_mp3_path=f"c{index}/meeting.mp3",
                        processing_wav_path=f"c{index}/processing.wav",
                    )
                )
            await session.commit()

    asyncio.run(seed())

    state = {"running": 0, "max_running": 0}

    async def fake_process(_session, _meeting, _settings) -> None:
        state["running"] += 1
        state["max_running"] = max(state["max_running"], state["running"])
        await asyncio.sleep(0.2)
        state["running"] -= 1

    monkeypatch.setattr("app.worker.process_meeting", fake_process)

    async def run() -> None:
        task = asyncio.create_task(worker_loop(settings))
        try:
            for _ in range(300):
                await asyncio.sleep(0.01)
                if state["max_running"] >= 2:
                    break
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())
    asyncio.run(seed_db.dispose())

    assert state["max_running"] == 2
