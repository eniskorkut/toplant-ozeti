"""Meeting timestamps must carry an explicit UTC offset so browsers show local time."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import Meeting


@pytest.fixture
def context(tmp_path: Path) -> Iterator[tuple[TestClient, Database]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'times.sqlite'}")
    asyncio.run(database.init())
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    settings = Settings(data_dir=meetings_dir, database_url=database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def test_created_at_is_serialized_as_utc_with_offset(context) -> None:
    client, database = context
    moment = datetime(2026, 9, 19, 11, 5, 40, 494864, tzinfo=UTC)

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="tz1",
                    status="uploaded",
                    created_at=moment,
                    duration_seconds=1.0,
                    audio_mp3_path="tz1/meeting.mp3",
                    processing_wav_path="tz1/processing.wav",
                )
            )
            await session.commit()

    asyncio.run(seed())

    status = client.get("/api/v1/meetings/tz1").json()
    listed = client.get("/api/v1/meetings").json()["meetings"][0]

    for value in (status["created_at"], listed["created_at"]):
        parsed = datetime.fromisoformat(value)
        assert parsed.tzinfo is not None, "offset missing: browsers would assume local time"
        assert parsed.astimezone(UTC) == moment
        # Türkiye is UTC+3: the same instant must render as 14:05 local, not 11:05.
        assert (parsed + timedelta(hours=3)).hour == 14
