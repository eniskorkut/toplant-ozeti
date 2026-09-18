"""Tests for the meeting list, MP3 streaming endpoint and JSON-mode portability."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import (
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_QUEUED,
    Meeting,
    MeetingAnalysis,
    TranscriptTurn,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Database, Settings]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'list.sqlite'}")
    asyncio.run(database.init())
    meetings_dir = tmp_path / "meetings"
    test_settings = Settings(data_dir=meetings_dir, database_url=database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database, test_settings
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def seed_meeting(
    database: Database,
    settings: Settings,
    meeting_id: str,
    *,
    status: str = MEETING_STATUS_COMPLETED,
    duration: float = 12.5,
    audio_bytes: bytes | None = None,
    audio_path: str | None = None,
    turns: int = 0,
    analysis_status: str | None = None,
) -> None:
    recording_dir = settings.meetings_dir / meeting_id
    recording_dir.mkdir(parents=True, exist_ok=True)
    if audio_bytes is not None:
        (recording_dir / "meeting.mp3").write_bytes(audio_bytes)

    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status=status,
                    duration_seconds=duration,
                    audio_mp3_path=audio_path or f"{meeting_id}/meeting.mp3",
                    processing_wav_path=f"{meeting_id}/processing.wav",
                )
            )
            for ordinal in range(turns):
                session.add(
                    TranscriptTurn(
                        meeting_id=meeting_id,
                        ordinal=ordinal,
                        speaker="Kişi 1",
                        start_seconds=float(ordinal),
                        end_seconds=float(ordinal) + 0.5,
                        text=f"turn {ordinal}",
                    )
                )
            if analysis_status is not None:
                session.add(
                    MeetingAnalysis(meeting_id=meeting_id, status=analysis_status)
                )
            await session.commit()

    asyncio.run(run())


def test_list_returns_newest_first_without_transcript_text(client) -> None:
    test_client, database, settings = client
    seed_meeting(database, settings, "older", turns=2)
    seed_meeting(
        database,
        settings,
        "newer",
        status=MEETING_STATUS_QUEUED,
        analysis_status=ANALYSIS_STATUS_QUEUED,
    )

    body = test_client.get("/api/v1/meetings").json()

    assert body["count"] == 2
    assert [item["meeting_id"] for item in body["meetings"]] == ["newer", "older"]
    older = body["meetings"][1]
    assert older["has_transcript"] is True
    assert older["analysis_status"] is None
    assert "text" not in older
    newer = body["meetings"][0]
    assert newer["has_transcript"] is False
    assert newer["analysis_status"] == ANALYSIS_STATUS_QUEUED
    assert newer["status"] == MEETING_STATUS_QUEUED


def test_list_pagination(client) -> None:
    test_client, database, settings = client
    for index in range(3):
        seed_meeting(database, settings, f"m{index}")

    body = test_client.get("/api/v1/meetings?limit=2&offset=1").json()

    assert body["count"] == 2
    assert len(body["meetings"]) == 2


def test_audio_endpoint_serves_mp3(client) -> None:
    test_client, database, settings = client
    payload = b"ID3fake-mp3-bytes" * 10
    seed_meeting(database, settings, "audio1", audio_bytes=payload)

    response = test_client.get("/api/v1/meetings/audio1/audio")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content == payload


def test_audio_endpoint_supports_range_requests(client) -> None:
    test_client, database, settings = client
    seed_meeting(database, settings, "audio2", audio_bytes=b"0123456789")

    response = test_client.get(
        "/api/v1/meetings/audio2/audio", headers={"Range": "bytes=2-5"}
    )

    assert response.status_code == 206
    assert response.content == b"2345"


def test_audio_endpoint_404_when_missing(client) -> None:
    test_client, database, settings = client
    seed_meeting(database, settings, "audio3", audio_bytes=None)

    assert test_client.get("/api/v1/meetings/audio3/audio").status_code == 404


def test_audio_endpoint_rejects_paths_outside_the_data_dir(client) -> None:
    test_client, database, settings = client
    # Simulate a tampered database row: the endpoint must never follow it out of
    # the meetings directory.
    seed_meeting(
        database,
        settings,
        "audio4",
        audio_bytes=None,
        audio_path="../../../../etc/hostname",
    )

    assert test_client.get("/api/v1/meetings/audio4/audio").status_code == 404


def test_audio_endpoint_404_for_unknown_meeting(client) -> None:
    test_client, _, _ = client
    assert test_client.get("/api/v1/meetings/missing/audio").status_code == 404


# --- JSON-mode portability -------------------------------------------------


class FakeResponse:
    def __init__(self) -> None:
        self.status_code = 200

    def json(self) -> dict:
        return {"choices": [{"message": {"content": json.dumps({"summary": "ok"})}}]}


def provider_payload(monkeypatch: pytest.MonkeyPatch, json_mode: bool) -> dict:
    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured.update(json)
        return FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    settings = Settings(
        llm_base_url="https://api.example.com/v1",
        llm_api_key="not-a-real-key",
        llm_model="test-model",
        llm_json_mode=json_mode,
    )
    from app.services.llm_provider import OpenAICompatibleProvider

    OpenAICompatibleProvider(settings).analyze(
        system_prompt="s", user_prompt="u", session_id="m1"
    )
    return captured


def test_response_format_is_omitted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = provider_payload(monkeypatch, json_mode=False)
    assert "response_format" not in payload


def test_response_format_is_sent_when_json_mode_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = provider_payload(monkeypatch, json_mode=True)
    assert payload["response_format"] == {"type": "json_object"}
