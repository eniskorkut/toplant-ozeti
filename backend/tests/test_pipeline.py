"""Pipeline state-machine tests with mocked ML adapters (no models are loaded)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import Settings
from app.db import Database
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    Meeting,
    TranscriptTurn,
)
from app.services import pipeline as pipeline_module
from app.services.diarization import DiarizationError, DiarizationResult, DiarizationSegment
from app.services.stt import SttError, SttResult, SttWord


def make_settings(tmp_path: Path) -> Settings:
    database_url = f"sqlite+aiosqlite:///{tmp_path / 'db.sqlite'}"
    return Settings(data_dir=tmp_path / "meetings", database_url=database_url)


def create_meeting(
    settings: Settings, meeting_id: str = "abc123", *, audio: bool = True
) -> Meeting:
    recording_dir = settings.meetings_dir / meeting_id
    recording_dir.mkdir(parents=True, exist_ok=True)
    if audio:
        # Minimal valid 16 kHz mono PCM WAV (1 s of silence) for the duration check.
        import wave

        with wave.open(str(recording_dir / "processing.wav"), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 16000)
    return Meeting(
        id=meeting_id,
        status=MEETING_STATUS_QUEUED,
        audio_mp3_path=f"{meeting_id}/meeting.mp3",
        processing_wav_path=f"{meeting_id}/processing.wav",
    )


def fake_stt_result() -> SttResult:
    return SttResult(
        words=[
            SttWord(0.0, 0.4, "Merhaba"),
            SttWord(0.5, 0.9, "dunya"),
            SttWord(5.0, 5.4, "kayip"),
        ],
        segments=[],
        text="Merhaba dunya kayip",
        language="tr",
        inference_seconds=1.5,
    )


def fake_diarization_result() -> DiarizationResult:
    return DiarizationResult(
        segments=[DiarizationSegment(0.0, 1.0, "cluster_5")],
        num_speakers=1,
        inference_seconds=0.5,
        model_load_seconds=0.1,
    )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'pipeline.sqlite'}")
    asyncio.run(db.init())
    return db


def test_queued_to_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, database: Database
) -> None:
    settings = make_settings(tmp_path)
    monkeypatch.setattr(
        "app.services.providers.local.transcribe", lambda path, settings: fake_stt_result()
    )
    monkeypatch.setattr(
        "app.services.providers.local.diarize",
        lambda path, settings, **kw: fake_diarization_result(),
    )

    async def run() -> tuple[str, list[TranscriptTurn]]:
        async with database.session_factory() as session:
            session.add(create_meeting(settings))
            await session.commit()
            meeting = await pipeline_module.claim_next_meeting(session)
            assert meeting is not None
            assert meeting.status == MEETING_STATUS_PROCESSING
            await pipeline_module.process_meeting(session, meeting, settings)
            turns = (
                (await session.execute(select(TranscriptTurn).order_by(TranscriptTurn.ordinal)))
                .scalars()
                .all()
            )
            refreshed = await session.get(Meeting, meeting.id)
            return refreshed.status, list(turns)

    status, turns = asyncio.run(run())
    assert status == MEETING_STATUS_COMPLETED
    assert [turn.speaker for turn in turns] == ["Kişi 1", "Bilinmeyen"]
    assert turns[0].text == "Merhaba dunya"
    assert turns[1].text == "kayip"
    assert turns[0].start_seconds == 0.0
    assert turns[0].end_seconds == 0.9


def test_queued_to_failed_removes_partial_turns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, database: Database
) -> None:
    settings = make_settings(tmp_path)

    def failing_diarize(path, settings, **kwargs):
        raise DiarizationError("simulated diarization failure")

    monkeypatch.setattr(
        "app.services.providers.local.transcribe", lambda path, settings: fake_stt_result()
    )
    monkeypatch.setattr("app.services.providers.local.diarize", failing_diarize)

    async def run() -> tuple[str, str | None, int]:
        async with database.session_factory() as session:
            session.add(create_meeting(settings))
            await session.commit()
            meeting = await pipeline_module.claim_next_meeting(session)
            assert meeting is not None
            await pipeline_module.process_meeting(session, meeting, settings)
            refreshed = await session.get(Meeting, meeting.id)
            turns = (await session.execute(select(TranscriptTurn))).scalars().all()
            return refreshed.status, refreshed.processing_error, len(turns)

    status, error, turn_count = asyncio.run(run())
    assert status == MEETING_STATUS_FAILED
    assert error is not None and "simulated diarization failure" in error
    assert turn_count == 0


def test_missing_model_fails_with_clear_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, database: Database
) -> None:
    settings = make_settings(tmp_path)

    def failing_stt(path, settings):
        raise SttError("whisper.cpp model not found: /models/whisper/missing.bin")

    monkeypatch.setattr("app.services.providers.local.transcribe", failing_stt)

    async def run() -> tuple[str, str | None]:
        async with database.session_factory() as session:
            session.add(create_meeting(settings, "missing-model"))
            await session.commit()
            meeting = await pipeline_module.claim_next_meeting(session)
            await pipeline_module.process_meeting(session, meeting, settings)
            refreshed = await session.get(Meeting, "missing-model")
            return refreshed.status, refreshed.processing_error

    status, error = asyncio.run(run())
    assert status == MEETING_STATUS_FAILED
    assert "model not found" in (error or "")


def test_missing_audio_fails(tmp_path: Path, database: Database) -> None:
    settings = make_settings(tmp_path)

    async def run() -> tuple[str, str | None]:
        async with database.session_factory() as session:
            session.add(create_meeting(settings, "no-audio", audio=False))
            await session.commit()
            meeting = await pipeline_module.claim_next_meeting(session)
            await pipeline_module.process_meeting(session, meeting, settings)
            refreshed = await session.get(Meeting, "no-audio")
            return refreshed.status, refreshed.processing_error

    status, error = asyncio.run(run())
    assert status == MEETING_STATUS_FAILED
    assert "processing audio missing" in (error or "")


def test_duplicate_claims_are_rejected(tmp_path: Path, database: Database) -> None:
    settings = make_settings(tmp_path)

    async def run() -> tuple[Meeting | None, Meeting | None]:
        async with database.session_factory() as session:
            session.add(create_meeting(settings, "dup"))
            await session.commit()
            first = await pipeline_module.claim_next_meeting(session)
            second = await pipeline_module.claim_next_meeting(session)
            return first, second

    first, second = asyncio.run(run())
    assert first is not None and first.id == "dup"
    assert second is None


def test_known_speaker_count_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, database: Database
) -> None:
    settings = make_settings(tmp_path)
    captured: dict = {}

    def recording_diarize(path, settings, *, requested_speaker_count=None):
        captured["requested"] = requested_speaker_count
        return fake_diarization_result()

    monkeypatch.setattr(
        "app.services.providers.local.transcribe", lambda path, settings: fake_stt_result()
    )
    monkeypatch.setattr("app.services.providers.local.diarize", recording_diarize)

    async def run() -> None:
        async with database.session_factory() as session:
            meeting = create_meeting(settings, "known-count")
            meeting.requested_speaker_count = 3
            session.add(meeting)
            await session.commit()
            claimed = await pipeline_module.claim_next_meeting(session)
            await pipeline_module.process_meeting(session, claimed, settings)

    asyncio.run(run())
    assert captured["requested"] == 3
