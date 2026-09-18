"""Tests for DELETE /api/v1/meetings/{id}: safety, shared artifacts, 409/404."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_QUEUED,
    MEETING_STATUS_UPLOADED,
    Meeting,
    MeetingAnalysis,
    TranscriptTurn,
)


@pytest.fixture
def context(tmp_path: Path) -> Iterator[tuple[TestClient, Database, Settings]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'delete.sqlite'}")
    asyncio.run(database.init())
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    settings = Settings(data_dir=meetings_dir, database_url=database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database, settings
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def seed_meeting(
    database: Database,
    settings: Settings,
    meeting_id: str,
    *,
    status: str = MEETING_STATUS_COMPLETED,
    audio_mp3_path: str | None = None,
    processing_wav_path: str | None = None,
    write_audio: bool = True,
    write_wav: bool = True,
    turns: int = 2,
    analysis_status: str | None = None,
) -> None:
    audio_mp3_path = audio_mp3_path or f"{meeting_id}/meeting.mp3"
    processing_wav_path = processing_wav_path or f"{meeting_id}/processing.wav"

    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status=status,
                    duration_seconds=22.0,
                    audio_mp3_path=audio_mp3_path,
                    processing_wav_path=processing_wav_path,
                )
            )
            for ordinal in range(turns):
                session.add(
                    TranscriptTurn(
                        meeting_id=meeting_id,
                        ordinal=ordinal,
                        speaker="Kişi 1",
                        start_seconds=float(ordinal),
                        end_seconds=float(ordinal) + 1.0,
                        text=f"turn {ordinal}",
                    )
                )
            if analysis_status is not None:
                session.add(
                    MeetingAnalysis(
                        meeting_id=meeting_id,
                        status=analysis_status,
                        summary="özet",
                    )
                )
            await session.commit()

    asyncio.run(run())

    if write_audio:
        audio = settings.meetings_dir / audio_mp3_path
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b"ID3fake-mp3")
    if write_wav:
        wav = settings.meetings_dir / processing_wav_path
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFFfake-wav")


def db_counts(database: Database, meeting_id: str) -> tuple[int, int, int]:
    async def run() -> tuple[int, int, int]:
        async with database.session_factory() as session:
            meetings = await session.scalar(
                select(func.count()).select_from(Meeting).where(Meeting.id == meeting_id)
            )
            turns = await session.scalar(
                select(func.count())
                .select_from(TranscriptTurn)
                .where(TranscriptTurn.meeting_id == meeting_id)
            )
            analyses = await session.scalar(
                select(func.count())
                .select_from(MeetingAnalysis)
                .where(MeetingAnalysis.meeting_id == meeting_id)
            )
            return int(meetings or 0), int(turns or 0), int(analyses or 0)

    return asyncio.run(run())


def test_delete_completed_meeting_removes_rows_and_unreferenced_artifacts(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "aaaa1111", analysis_status=ANALYSIS_STATUS_COMPLETED)

    response = client.delete("/api/v1/meetings/aaaa1111")

    assert response.status_code == 204
    assert db_counts(database, "aaaa1111") == (0, 0, 0)
    assert client.get("/api/v1/meetings/aaaa1111").status_code == 404
    assert not (settings.meetings_dir / "aaaa1111").exists()


def test_uploaded_and_failed_meetings_are_deletable(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "up000000", status=MEETING_STATUS_UPLOADED, turns=0)
    seed_meeting(
        database, settings, "fa000000", status=MEETING_STATUS_FAILED, turns=0, write_wav=False
    )

    assert client.delete("/api/v1/meetings/up000000").status_code == 204
    assert client.delete("/api/v1/meetings/fa000000").status_code == 204


@pytest.mark.parametrize(
    "meeting_status",
    [MEETING_STATUS_QUEUED, MEETING_STATUS_PROCESSING],
)
def test_active_meeting_status_returns_409(context, meeting_status: str) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "act00000", status=meeting_status)

    response = client.delete("/api/v1/meetings/act00000")

    assert response.status_code == 409
    assert db_counts(database, "act00000")[0] == 1


@pytest.mark.parametrize(
    "analysis_status",
    [ANALYSIS_STATUS_QUEUED, ANALYSIS_STATUS_PROCESSING],
)
def test_active_analysis_returns_409(context, analysis_status: str) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "ana00000", analysis_status=analysis_status)

    response = client.delete("/api/v1/meetings/ana00000")

    assert response.status_code == 409
    assert db_counts(database, "ana00000")[0] == 1


def test_missing_meeting_returns_404(context) -> None:
    client, _, _ = context
    assert client.delete("/api/v1/meetings/doesnotexist").status_code == 404


def test_shared_mp3_is_kept_while_another_meeting_references_it(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "source01")
    seed_meeting(
        database,
        settings,
        "comparison01",
        audio_mp3_path="source01/meeting.mp3",
        processing_wav_path="source01/processing.wav",
        write_audio=False,
        write_wav=False,
    )

    assert client.delete("/api/v1/meetings/comparison01").status_code == 204

    assert (settings.meetings_dir / "source01" / "meeting.mp3").is_file()
    assert (settings.meetings_dir / "source01" / "processing.wav").is_file()
    assert client.get("/api/v1/meetings/source01/audio").status_code == 200


def test_deleting_source_keeps_shared_artifacts_for_comparison(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "source02")
    seed_meeting(
        database,
        settings,
        "comparison02",
        audio_mp3_path="source02/meeting.mp3",
        processing_wav_path="source02/processing.wav",
        write_audio=False,
        write_wav=False,
    )

    assert client.delete("/api/v1/meetings/source02").status_code == 204

    assert (settings.meetings_dir / "source02" / "meeting.mp3").is_file()
    assert client.get("/api/v1/meetings/comparison02/audio").status_code == 200


def test_last_reference_deletes_the_artifact(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "source03")
    seed_meeting(
        database,
        settings,
        "comparison03",
        audio_mp3_path="source03/meeting.mp3",
        processing_wav_path="source03/processing.wav",
        write_audio=False,
        write_wav=False,
    )

    assert client.delete("/api/v1/meetings/source03").status_code == 204
    assert (settings.meetings_dir / "source03" / "meeting.mp3").is_file()

    assert client.delete("/api/v1/meetings/comparison03").status_code == 204
    assert not (settings.meetings_dir / "source03" / "meeting.mp3").exists()
    assert not (settings.meetings_dir / "source03" / "processing.wav").exists()
    assert not (settings.meetings_dir / "source03").exists()


def test_relative_path_escape_is_never_deleted(context, tmp_path: Path) -> None:
    client, database, settings = context
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"do not delete")
    seed_meeting(
        database,
        settings,
        "evil0001",
        audio_mp3_path="../outside.mp3",
        processing_wav_path="evil0001/processing.wav",
        write_audio=False,
    )

    assert client.delete("/api/v1/meetings/evil0001").status_code == 204
    assert outside.is_file()
    assert db_counts(database, "evil0001")[0] == 0


def test_absolute_path_is_never_deleted(context, tmp_path: Path) -> None:
    client, database, settings = context
    outside = tmp_path / "absolute.mp3"
    outside.write_bytes(b"do not delete")
    seed_meeting(
        database,
        settings,
        "evil0002",
        audio_mp3_path=str(outside),
        processing_wav_path="evil0002/processing.wav",
        write_audio=False,
    )

    assert client.delete("/api/v1/meetings/evil0002").status_code == 204
    assert outside.is_file()


def test_cleanup_failure_keeps_database_consistent(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "fail0001", write_audio=False, write_wav=True)
    # A directory where a file is expected makes unlink fail with OSError.
    (settings.meetings_dir / "fail0001" / "meeting.mp3").mkdir()

    response = client.delete("/api/v1/meetings/fail0001")

    assert response.status_code == 204
    assert db_counts(database, "fail0001") == (0, 0, 0)
    assert client.get("/api/v1/meetings/fail0001").status_code == 404


def test_delete_does_not_touch_other_meetings_artifacts(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "keep0001")
    seed_meeting(database, settings, "drop0001")

    assert client.delete("/api/v1/meetings/drop0001").status_code == 204

    assert (settings.meetings_dir / "keep0001" / "meeting.mp3").is_file()
    assert (settings.meetings_dir / "keep0001" / "processing.wav").is_file()
    assert client.get("/api/v1/meetings/keep0001/audio").status_code == 200


def test_deleted_meeting_disappears_from_the_list(context) -> None:
    client, database, settings = context
    seed_meeting(database, settings, "gone0001")
    seed_meeting(database, settings, "stay0001")

    assert client.delete("/api/v1/meetings/gone0001").status_code == 204

    payload = client.get("/api/v1/meetings").json()
    ids = [row["meeting_id"] for row in payload["meetings"]]
    assert "gone0001" not in ids
    assert "stay0001" in ids
