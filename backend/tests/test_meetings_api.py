"""API tests for the meeting processing endpoints (DB + mocked pipeline boundary)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    UNKNOWN_SPEAKER,
    Meeting,
    TranscriptTurn,
)


@pytest.fixture
def meeting_client(tmp_path: Path) -> Iterator[tuple[TestClient, Database, Settings]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite'}")
    asyncio.run(database.init())
    settings = Settings(data_dir=tmp_path / "meetings", database_url=database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database, settings
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def add_meeting(database: Database, meeting_id: str, status: str = "uploaded", **kwargs) -> None:
    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status=status,
                    audio_mp3_path=f"{meeting_id}/meeting.mp3",
                    processing_wav_path=f"{meeting_id}/processing.wav",
                    **kwargs,
                )
            )
            await session.commit()

    asyncio.run(run())


def test_process_queues_uploaded_meeting(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m1")

    response = client.post("/api/v1/meetings/m1/process", json={"speaker_count": None})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == MEETING_STATUS_QUEUED
    assert body["meeting_id"] == "m1"
    assert body["has_transcript"] is False


def test_process_accepts_known_speaker_count(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m2")

    response = client.post("/api/v1/meetings/m2/process", json={"speaker_count": 3})

    assert response.status_code == 202
    assert response.json()["requested_speaker_count"] == 3


def test_process_rejects_invalid_speaker_count(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m3")

    assert client.post("/api/v1/meetings/m3/process", json={"speaker_count": 0}).status_code == 422
    assert client.post("/api/v1/meetings/m3/process", json={"speaker_count": 99}).status_code == 422


def test_process_is_idempotent_while_queued_or_completed(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m4")

    first = client.post("/api/v1/meetings/m4/process", json={})
    second = client.post("/api/v1/meetings/m4/process", json={"speaker_count": 2})

    assert first.status_code == second.status_code == 202
    # Still queued: the second request must not overwrite the queued job.
    assert second.json()["status"] == MEETING_STATUS_QUEUED
    assert second.json()["requested_speaker_count"] is None


def test_process_completed_meeting_is_noop(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m5", status=MEETING_STATUS_COMPLETED)

    response = client.post("/api/v1/meetings/m5/process", json={})

    assert response.status_code == 202
    assert response.json()["status"] == MEETING_STATUS_COMPLETED


def test_process_missing_meeting_404(meeting_client) -> None:
    client, _, _ = meeting_client
    assert client.post("/api/v1/meetings/nope/process", json={}).status_code == 404


def test_status_endpoint_reports_processing(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m6", status=MEETING_STATUS_PROCESSING)

    body = client.get("/api/v1/meetings/m6").json()

    assert body["status"] == MEETING_STATUS_PROCESSING
    assert body["has_transcript"] is False


def test_status_endpoint_reports_failure_message(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(
        database,
        "m7",
        status=MEETING_STATUS_FAILED,
        processing_error="simulated failure",
    )

    body = client.get("/api/v1/meetings/m7").json()

    assert body["status"] == MEETING_STATUS_FAILED
    assert body["processing_error"] == "simulated failure"


def test_transcript_conflict_before_completion(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m8", status=MEETING_STATUS_QUEUED)

    response = client.get("/api/v1/meetings/m8/transcript")

    assert response.status_code == 409
    assert "not available" in response.json()["detail"]


def test_transcript_returns_turns_and_speakers(meeting_client) -> None:
    client, database, _ = meeting_client
    add_meeting(database, "m9", status=MEETING_STATUS_COMPLETED)

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add_all(
                [
                    TranscriptTurn(
                        meeting_id="m9",
                        ordinal=0,
                        speaker="Kişi 1",
                        start_seconds=0.0,
                        end_seconds=2.0,
                        text="Merhaba",
                    ),
                    TranscriptTurn(
                        meeting_id="m9",
                        ordinal=1,
                        speaker=UNKNOWN_SPEAKER,
                        start_seconds=2.5,
                        end_seconds=3.0,
                        text="kayip",
                    ),
                    TranscriptTurn(
                        meeting_id="m9",
                        ordinal=2,
                        speaker="Kişi 2",
                        start_seconds=3.0,
                        end_seconds=4.0,
                        text="Tamam",
                    ),
                ]
            )
            await session.commit()

    asyncio.run(seed())

    body = client.get("/api/v1/meetings/m9/transcript").json()

    assert body["speakers"] == ["Kişi 1", "Kişi 2"]
    assert body["unresolved_label"] == UNKNOWN_SPEAKER
    assert body["unresolved_turns"] == 1
    assert [turn["ordinal"] for turn in body["turns"]] == [0, 1, 2]
    assert body["turns"][1]["text"] == "kayip"
