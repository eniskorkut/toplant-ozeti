"""Per-meeting provider selection, capability endpoint and persistence tests."""

from __future__ import annotations

import asyncio
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_QUEUED,
    Meeting,
    TranscriptTurn,
)


def local_only_settings(tmp_path: Path, database_url: str) -> Settings:
    return Settings(
        data_dir=tmp_path / "meetings",
        database_url=database_url,
        transcription_provider="local",
        llm_provider="openai_compatible",
        llm_base_url=None,
        llm_api_key=None,
        llm_model=None,
        elevenlabs_api_key=None,
    )


def cloud_settings(tmp_path: Path, database_url: str) -> Settings:
    values = local_only_settings(tmp_path, database_url).model_dump()
    values["elevenlabs_api_key"] = "sk_test_key_value"
    return Settings(**values)


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, Database]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'providers.sqlite'}")
    asyncio.run(database.init())
    settings = local_only_settings(tmp_path, database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def override_settings(settings: Settings) -> None:
    app.dependency_overrides[get_settings] = lambda: settings


def add_meeting(database: Database, meeting_id: str, status: str = "uploaded") -> None:
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


def meeting_row(database: Database, meeting_id: str) -> Meeting:
    async def run() -> Meeting:
        async with database.session_factory() as session:
            return await session.get(Meeting, meeting_id)

    return asyncio.run(run())


# --- process request -------------------------------------------------------


def test_omitted_provider_keeps_the_server_default(api, tmp_path: Path) -> None:
    client, database = api
    add_meeting(database, "m-default")

    response = client.post("/api/v1/meetings/m-default/process", json={})

    assert response.status_code == 202
    assert meeting_row(database, "m-default").requested_transcription_provider is None


def test_explicit_local_provider_is_accepted(api) -> None:
    client, database = api
    add_meeting(database, "m-local")

    response = client.post(
        "/api/v1/meetings/m-local/process", json={"transcription_provider": "local"}
    )

    assert response.status_code == 202
    assert meeting_row(database, "m-local").requested_transcription_provider == "local"


def test_explicit_elevenlabs_requires_configuration(api, tmp_path: Path) -> None:
    client, database = api
    add_meeting(database, "m-cloud-unconfigured")

    response = client.post(
        "/api/v1/meetings/m-cloud-unconfigured/process",
        json={"transcription_provider": "elevenlabs"},
    )

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    assert meeting_row(database, "m-cloud-unconfigured").status == "uploaded"


def test_explicit_elevenlabs_accepted_when_configured(api, tmp_path: Path) -> None:
    client, database = api
    override_settings(cloud_settings(tmp_path, database.url))
    add_meeting(database, "m-cloud")

    response = client.post(
        "/api/v1/meetings/m-cloud/process", json={"transcription_provider": "elevenlabs"}
    )

    assert response.status_code == 202
    assert meeting_row(database, "m-cloud").requested_transcription_provider == "elevenlabs"
    assert meeting_row(database, "m-cloud").transcription_provider is None  # not yet produced


def test_unsupported_provider_is_rejected(api) -> None:
    client, database = api
    add_meeting(database, "m-bogus")

    response = client.post(
        "/api/v1/meetings/m-bogus/process", json={"transcription_provider": "openai"}
    )

    assert response.status_code == 422


# --- capabilities ----------------------------------------------------------


def test_capability_endpoint_reports_local_available(api) -> None:
    client, _ = api
    body = client.get("/api/v1/transcription/providers").json()

    local = next(item for item in body["providers"] if item["id"] == "local")
    assert local == {"id": "local", "available": True, "cloud": False, "label": "Yerel"}
    assert body["default"] == "local"


def test_capability_endpoint_reports_elevenlabs_unavailable_without_key(api) -> None:
    client, _ = api
    body = client.get("/api/v1/transcription/providers").json()

    elevenlabs = next(item for item in body["providers"] if item["id"] == "elevenlabs")
    assert elevenlabs["available"] is False
    assert elevenlabs["cloud"] is True
    assert elevenlabs["label"] == "ElevenLabs"


def test_capability_endpoint_reports_elevenlabs_available_with_key(api, tmp_path: Path) -> None:
    client, database = api
    override_settings(cloud_settings(tmp_path, database.url))

    body = client.get("/api/v1/transcription/providers").json()

    elevenlabs = next(item for item in body["providers"] if item["id"] == "elevenlabs")
    assert elevenlabs["available"] is True


def test_capability_response_contains_no_secret(api, tmp_path: Path) -> None:
    client, database = api
    override_settings(cloud_settings(tmp_path, database.url))

    body = client.get("/api/v1/transcription/providers")
    text = body.text

    assert "sk_test_key_value" not in text
    assert "sk_" not in text
    assert "api_key" not in text.lower()
    assert "authorization" not in text.lower()


# --- persistence through the worker path -----------------------------------


def test_requested_and_actual_provider_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = cloud_settings(tmp_path, "sqlite+aiosqlite:///" + str(tmp_path / "persist.sqlite"))
    database = Database(settings.database_url)
    asyncio.run(database.init())

    recording_dir = settings.meetings_dir / "m-ok"
    recording_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(recording_dir / "processing.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)

    captured: dict = {}

    def fake_transcribe(self, audio_path, *, requested_speaker_count, settings):
        from app.services.transcription import NormalizedWord, TranscriptionResult

        captured["provider"] = self.name
        captured["settings_provider"] = settings.transcription_provider
        return TranscriptionResult(
            text="Merhaba",
            words=[NormalizedWord(0.1, 0.5, "Merhaba", "speaker_a")],
            language="tur",
            latency_seconds=0.5,
            provider="elevenlabs",
            model="scribe_v2",
        )

    monkeypatch.setattr(
        "app.services.providers.elevenlabs.ElevenLabsProvider.transcribe", fake_transcribe
    )

    from app.services.pipeline import claim_next_meeting, process_meeting

    async def run() -> Meeting:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="m-ok",
                    status=MEETING_STATUS_QUEUED,
                    audio_mp3_path="m-ok/meeting.mp3",
                    processing_wav_path="m-ok/processing.wav",
                    requested_transcription_provider="elevenlabs",
                )
            )
            await session.commit()
            meeting = await claim_next_meeting(session)
            await process_meeting(session, meeting, settings)
            return await session.get(Meeting, "m-ok")

    meeting = asyncio.run(run())

    assert captured["provider"] == "elevenlabs"
    assert captured["settings_provider"] == "elevenlabs"  # override applied
    assert meeting.status == MEETING_STATUS_COMPLETED
    assert meeting.requested_transcription_provider == "elevenlabs"
    assert meeting.transcription_provider == "elevenlabs"
    assert meeting.transcription_model == "scribe_v2"


def test_local_override_wins_over_cloud_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = cloud_settings(tmp_path, "sqlite+aiosqlite:///" + str(tmp_path / "local.sqlite"))
    settings = settings.model_copy(update={"transcription_provider": "elevenlabs"})
    database = Database(settings.database_url)
    asyncio.run(database.init())

    recording_dir = settings.meetings_dir / "m-local"
    recording_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(recording_dir / "processing.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)

    captured: dict = {}

    def fake_local_transcribe(self, audio_path, *, requested_speaker_count, settings):
        from app.services.transcription import NormalizedWord, TranscriptionResult

        captured["provider"] = self.name
        return TranscriptionResult(
            text="Merhaba",
            words=[NormalizedWord(0.1, 0.5, "Merhaba", "cluster_1")],
            language="tr",
            latency_seconds=1.0,
            provider="local",
            model="whisper-large-v3-turbo-q8",
        )

    monkeypatch.setattr(
        "app.services.providers.local.LocalProvider.transcribe", fake_local_transcribe
    )
    monkeypatch.setattr(
        "app.services.providers.elevenlabs.ElevenLabsProvider.transcribe",
        lambda *args, **kwargs: pytest.fail("cloud provider must not be used"),
    )

    from app.services.pipeline import claim_next_meeting, process_meeting

    async def run() -> Meeting:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="m-local",
                    status=MEETING_STATUS_QUEUED,
                    audio_mp3_path="m-local/meeting.mp3",
                    processing_wav_path="m-local/processing.wav",
                    requested_transcription_provider="local",
                )
            )
            await session.commit()
            meeting = await claim_next_meeting(session)
            await process_meeting(session, meeting, settings)
            return await session.get(Meeting, "m-local")

    meeting = asyncio.run(run())

    assert captured["provider"] == "local"
    assert meeting.transcription_provider == "local"
    assert meeting.transcription_model == "whisper-large-v3-turbo-q8"


# --- legacy compatibility --------------------------------------------------


def test_meeting_detail_and_list_render_null_provider_metadata(api) -> None:
    client, database = api
    add_meeting(database, "legacy", status=MEETING_STATUS_COMPLETED)

    detail = client.get("/api/v1/meetings/legacy").json()
    listed = client.get("/api/v1/meetings").json()["meetings"][0]

    assert detail["transcription_provider"] is None
    assert detail["transcription_model"] is None
    assert listed["transcription_provider"] is None
    assert listed["transcription_model"] is None


def test_list_exposes_provider_metadata_without_transcript_text(
    monkeypatch: pytest.MonkeyPatch, api, tmp_path: Path
) -> None:
    client, database = api

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="with-provider",
                    status=MEETING_STATUS_COMPLETED,
                    audio_mp3_path="with-provider/meeting.mp3",
                    processing_wav_path="with-provider/processing.wav",
                    transcription_provider="elevenlabs",
                    transcription_model="scribe_v2",
                )
            )
            session.add(
                TranscriptTurn(
                    meeting_id="with-provider",
                    ordinal=0,
                    speaker="Kişi 1",
                    start_seconds=0.0,
                    end_seconds=1.0,
                    text="gizli metin",
                )
            )
            await session.commit()

    asyncio.run(seed())

    body = client.get("/api/v1/meetings").text
    payload = client.get("/api/v1/meetings").json()["meetings"][0]

    assert payload["transcription_provider"] == "elevenlabs"
    assert payload["transcription_model"] == "scribe_v2"
    assert "gizli metin" not in body
