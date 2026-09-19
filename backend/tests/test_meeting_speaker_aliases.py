"""Meeting-scoped speaker aliases: set/reset, isolation and cascade deletion."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import Meeting, MeetingSpeakerAlias


@pytest.fixture
def context(tmp_path: Path) -> Iterator[tuple[TestClient, Database]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'aliases.sqlite'}")
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


def seed_meeting(database: Database, meeting_id: str) -> None:
    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status="uploaded",
                    duration_seconds=1.0,
                    audio_mp3_path=f"{meeting_id}/meeting.mp3",
                    processing_wav_path=f"{meeting_id}/processing.wav",
                )
            )
            await session.commit()

    asyncio.run(run())


def test_meeting_alias_set_and_reset(context) -> None:
    client, database = context
    seed_meeting(database, "m1")

    url = "/api/v1/meetings/m1/speakers/Kişi 1/alias"
    assert client.put(url, json={"display_name": "  Ahmet  "}).json()["display_name"] == "Ahmet"
    assert client.put(url, json={"display_name": "Mehmet"}).status_code == 200
    assert client.get("/api/v1/meetings/m1/speakers").json()["aliases"] == {"Kişi 1": "Mehmet"}
    assert client.delete(url).status_code == 204
    assert client.get("/api/v1/meetings/m1/speakers").json()["aliases"] == {}


def test_meeting_alias_validation(context) -> None:
    client, database = context
    seed_meeting(database, "m2")
    url = "/api/v1/meetings/m2/speakers/Kişi 1/alias"

    assert client.put(url, json={"display_name": "   "}).status_code == 422
    assert client.put(url, json={"display_name": "a\x00b"}).status_code == 422
    assert client.put(url, json={"display_name": "x" * 51}).status_code == 422
    unresolved = client.put(
        "/api/v1/meetings/m2/speakers/Bilinmeyen/alias", json={"display_name": "A"}
    )
    assert unresolved.status_code == 422


def test_meeting_deletion_removes_aliases(context) -> None:
    client, database = context

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="del1",
                    status="uploaded",
                    duration_seconds=1.0,
                    audio_mp3_path="del1/meeting.mp3",
                    processing_wav_path="del1/processing.wav",
                )
            )
            session.add(
                MeetingSpeakerAlias(
                    meeting_id="del1", canonical_speaker="Kişi 1", display_name="Ahmet"
                )
            )
            await session.commit()

    asyncio.run(seed())
    assert client.delete("/api/v1/meetings/del1").status_code == 204

    async def alias_count() -> int:
        async with database.session_factory() as session:
            result = await session.execute(
                select(MeetingSpeakerAlias).where(MeetingSpeakerAlias.meeting_id == "del1")
            )
            return len(result.scalars().all())

    assert asyncio.run(alias_count()) == 0
