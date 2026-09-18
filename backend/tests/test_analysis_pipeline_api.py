"""Analysis pipeline, queue API, worker priority and the local mock E2E."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import (
    ANALYSIS_STATUS_COMPLETED,
    ANALYSIS_STATUS_FAILED,
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_QUEUED,
    Meeting,
    MeetingAnalysis,
    TranscriptTurn,
)
from app.services import analysis_pipeline
from app.services.llm_provider import MockProvider

MOCK_PAYLOAD = {
    "summary": "Özet.",
    "topics": ["konu"],
    "decisions": [{"text": "Karar.", "source_turn_ordinals": [1]}],
    "action_items": [
        {"task": "Görev.", "owner": "Kişi 2", "due_date_text": None, "source_turn_ordinals": [2]}
    ],
    "important_moments": [
        {"title": "An", "description": "Açıklama", "source_turn_ordinal": 0}
    ],
}


def settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path / "meetings",
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'analysis.sqlite'}",
        "llm_provider": "mock",
    }
    values.update(overrides)
    return Settings(**values)


def seed_completed_meeting(
    database: Database, meeting_id: str = "m1", *, turns: bool = True
) -> None:
    async def run() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=meeting_id,
                    status=MEETING_STATUS_COMPLETED,
                    audio_mp3_path=f"{meeting_id}/meeting.mp3",
                    processing_wav_path=f"{meeting_id}/processing.wav",
                    duration_seconds=12.0,
                )
            )
            if turns:
                session.add_all(
                    [
                        TranscriptTurn(
                            meeting_id=meeting_id,
                            ordinal=0,
                            speaker="Kişi 1",
                            start_seconds=0.0,
                            end_seconds=5.0,
                            text="Merhaba",
                        ),
                        TranscriptTurn(
                            meeting_id=meeting_id,
                            ordinal=1,
                            speaker="Kişi 2",
                            start_seconds=5.42,
                            end_seconds=8.0,
                            text="Tamam",
                        ),
                        TranscriptTurn(
                            meeting_id=meeting_id,
                            ordinal=2,
                            speaker="Kişi 2",
                            start_seconds=9.81,
                            end_seconds=12.0,
                            text="Görüşürüz",
                        ),
                    ]
                )
            await session.commit()

    asyncio.run(run())


def seed_queued_analysis(database: Database, meeting_id: str = "m1") -> None:
    async def run() -> None:
        async with database.session_factory() as session:
            session.add(MeetingAnalysis(meeting_id=meeting_id, status=ANALYSIS_STATUS_QUEUED))
            await session.commit()

    asyncio.run(run())


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Database, Settings]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'api.sqlite'}")
    asyncio.run(database.init())
    test_settings = settings(tmp_path, database_url=database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database, test_settings
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


# --- queue API -------------------------------------------------------------


def test_analyze_requires_completed_transcript(client) -> None:
    test_client, database, _ = client

    async def add_queued() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="mq",
                    status=MEETING_STATUS_QUEUED,
                    audio_mp3_path="mq/meeting.mp3",
                    processing_wav_path="mq/processing.wav",
                )
            )
            await session.commit()

    asyncio.run(add_queued())
    response = test_client.post("/api/v1/meetings/mq/analyze")
    assert response.status_code == 409
    assert "must be completed" in response.json()["detail"]


def test_analyze_queues_and_is_idempotent(client) -> None:
    test_client, database, _ = client
    seed_completed_meeting(database)

    first = test_client.post("/api/v1/meetings/m1/analyze")
    second = test_client.post("/api/v1/meetings/m1/analyze")

    assert first.status_code == second.status_code == 202
    assert first.json()["status"] == ANALYSIS_STATUS_QUEUED
    assert second.json()["status"] == ANALYSIS_STATUS_QUEUED

    async def count() -> int:
        async with database.session_factory() as session:
            rows = (await session.execute(select(MeetingAnalysis))).scalars().all()
            return len(rows)

    assert asyncio.run(count()) == 1


def test_analysis_get_404_before_queue(client) -> None:
    test_client, database, _ = client
    seed_completed_meeting(database)
    assert test_client.get("/api/v1/meetings/m1/analysis").status_code == 404


def test_failed_analysis_can_be_retried(client) -> None:
    test_client, database, _ = client
    seed_completed_meeting(database)

    async def seed_failed() -> None:
        async with database.session_factory() as session:
            session.add(
                MeetingAnalysis(
                    meeting_id="m1",
                    status=ANALYSIS_STATUS_FAILED,
                    analysis_error="previous failure",
                )
            )
            await session.commit()

    asyncio.run(seed_failed())

    response = test_client.post("/api/v1/meetings/m1/analyze")

    assert response.status_code == 202
    assert response.json()["status"] == ANALYSIS_STATUS_QUEUED
    assert response.json()["analysis_error"] is None


def test_completed_analysis_is_stable(client) -> None:
    test_client, database, _ = client
    seed_completed_meeting(database)

    async def seed_completed() -> None:
        async with database.session_factory() as session:
            session.add(
                MeetingAnalysis(
                    meeting_id="m1",
                    status=ANALYSIS_STATUS_COMPLETED,
                    summary="Mevcut özet",
                    topics_json=json.dumps(["t"]),
                    decisions_json="[]",
                    action_items_json="[]",
                    important_moments_json="[]",
                    provider="mock",
                    model="mock-model",
                )
            )
            await session.commit()

    asyncio.run(seed_completed())

    response = test_client.post("/api/v1/meetings/m1/analyze")

    assert response.status_code == 202
    assert response.json()["status"] == ANALYSIS_STATUS_COMPLETED
    assert response.json()["summary"] == "Mevcut özet"


# --- pipeline --------------------------------------------------------------


def test_pipeline_completes_with_mock_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_settings = settings(tmp_path)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())
    seed_completed_meeting(database)
    seed_queued_analysis(database)
    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: MockProvider())

    async def run() -> MeetingAnalysis:
        async with database.session_factory() as session:
            analysis = await analysis_pipeline.claim_next_analysis(session)
            assert analysis is not None
            await analysis_pipeline.process_analysis(session, analysis, test_settings)
            return await session.get(MeetingAnalysis, analysis.id)

    stored = asyncio.run(run())
    assert stored.status == ANALYSIS_STATUS_COMPLETED
    assert stored.summary == "Mock analysis summary."
    assert stored.provider == "mock"
    assert stored.repair_attempts == 0
    assert stored.input_chars and stored.input_chars > 0
    assert json.loads(stored.topics_json or "[]") == ["mock topic"]


def test_malformed_output_gets_one_repair_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_settings = settings(tmp_path)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())
    seed_completed_meeting(database)
    seed_queued_analysis(database)

    # First response references a nonexistent ordinal: schema-valid but semantically
    # invalid, which must trigger exactly one repair request.
    provider = MockProvider(
        content=json.dumps(
            {
                "summary": "İlk deneme.",
                "decisions": [{"text": "Karar", "source_turn_ordinals": [99]}],
            }
        )
    )

    class RepairingProvider:
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, *, system_prompt: str, user_prompt: str, session_id: str):
            self.calls += 1
            if self.calls == 1:
                return provider.analyze(
                    system_prompt=system_prompt, user_prompt=user_prompt, session_id=session_id
                )
            from app.services.llm_provider import ProviderResponse

            return ProviderResponse(
                content=json.dumps(MOCK_PAYLOAD),
                provider="mock",
                model="mock-model",
                latency_seconds=0.1,
            )

    repairing = RepairingProvider()
    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: repairing)

    async def run() -> MeetingAnalysis:
        async with database.session_factory() as session:
            analysis = await analysis_pipeline.claim_next_analysis(session)
            await analysis_pipeline.process_analysis(session, analysis, test_settings)
            return await session.get(MeetingAnalysis, analysis.id)

    stored = asyncio.run(run())
    assert repairing.calls == 2
    assert stored.status == ANALYSIS_STATUS_COMPLETED
    assert stored.repair_attempts == 1


def test_malformed_output_twice_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_settings = settings(tmp_path)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())
    seed_completed_meeting(database)
    seed_queued_analysis(database)
    monkeypatch.setattr(
        analysis_pipeline, "build_provider", lambda _settings: MockProvider(content="{broken")
    )

    async def run() -> MeetingAnalysis:
        async with database.session_factory() as session:
            analysis = await analysis_pipeline.claim_next_analysis(session)
            await analysis_pipeline.process_analysis(session, analysis, test_settings)
            return await session.get(MeetingAnalysis, analysis.id)

    stored = asyncio.run(run())
    assert stored.status == ANALYSIS_STATUS_FAILED
    assert stored.repair_attempts == 1
    assert stored.analysis_error


def test_long_transcript_is_rejected_without_truncation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_settings = settings(tmp_path, llm_max_transcript_chars=10)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())
    seed_completed_meeting(database)
    seed_queued_analysis(database)
    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: MockProvider())

    async def run() -> MeetingAnalysis:
        async with database.session_factory() as session:
            analysis = await analysis_pipeline.claim_next_analysis(session)
            await analysis_pipeline.process_analysis(session, analysis, test_settings)
            return await session.get(MeetingAnalysis, analysis.id)

    stored = asyncio.run(run())
    assert stored.status == ANALYSIS_STATUS_FAILED
    assert "transcript too large for single-pass analysis" in (stored.analysis_error or "")


def test_analysis_failure_leaves_transcript_intact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_settings = settings(tmp_path)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())
    seed_completed_meeting(database)
    seed_queued_analysis(database)

    class ExplodingProvider:
        def analyze(self, *, system_prompt: str, user_prompt: str, session_id: str):
            raise RuntimeError("boom")

    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: ExplodingProvider())

    async def run() -> tuple[MeetingAnalysis, list[TranscriptTurn]]:
        async with database.session_factory() as session:
            analysis = await analysis_pipeline.claim_next_analysis(session)
            await analysis_pipeline.process_analysis(session, analysis, test_settings)
            stored = await session.get(MeetingAnalysis, analysis.id)
            turns = (
                (await session.execute(select(TranscriptTurn).order_by(TranscriptTurn.ordinal)))
                .scalars()
                .all()
            )
            meeting = await session.get(Meeting, "m1")
            assert meeting.status == MEETING_STATUS_COMPLETED
            return stored, list(turns)

    stored, turns = asyncio.run(run())
    assert stored.status == ANALYSIS_STATUS_FAILED
    assert len(turns) == 3
    assert [turn.ordinal for turn in turns] == [0, 1, 2]


# --- worker ----------------------------------------------------------------


def test_worker_prioritizes_transcription_then_analysis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test_settings = settings(tmp_path)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())

    # A queued meeting (no audio file -> would fail) and a queued analysis.
    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="prio",
                    status=MEETING_STATUS_QUEUED,
                    audio_mp3_path="prio/meeting.mp3",
                    processing_wav_path="prio/processing.wav",
                )
            )
            session.add(
                Meeting(
                    id="done",
                    status=MEETING_STATUS_COMPLETED,
                    audio_mp3_path="done/meeting.mp3",
                    processing_wav_path="done/processing.wav",
                )
            )
            await session.commit()
            session.add(MeetingAnalysis(meeting_id="done", status=ANALYSIS_STATUS_QUEUED))
            await session.commit()

    asyncio.run(seed())

    # The worker imports the processors into its own namespace, so patch there.
    order: list[str] = []

    async def fake_process_meeting(*args, **kwargs):
        order.append("transcription")

    async def fake_process_analysis(*args, **kwargs):
        order.append("analysis")

    monkeypatch.setattr("app.worker.process_meeting", fake_process_meeting)
    monkeypatch.setattr("app.worker.process_analysis", fake_process_analysis)

    from app.worker import run_once

    async def run() -> None:
        await run_once(database, test_settings)
        await run_once(database, test_settings)

    asyncio.run(run())
    assert order == ["transcription", "analysis"]


def test_analysis_is_claimed_atomically(tmp_path: Path) -> None:
    test_settings = settings(tmp_path)
    database = Database(test_settings.database_url)
    asyncio.run(database.init())
    seed_completed_meeting(database)

    async def seed_analysis() -> None:
        async with database.session_factory() as session:
            session.add(MeetingAnalysis(meeting_id="m1", status=ANALYSIS_STATUS_QUEUED))
            await session.commit()

    asyncio.run(seed_analysis())

    async def run() -> tuple[MeetingAnalysis | None, MeetingAnalysis | None, str]:
        async with database.session_factory() as session:
            first = await analysis_pipeline.claim_next_analysis(session)
            second = await analysis_pipeline.claim_next_analysis(session)
            status = (await session.get(MeetingAnalysis, first.id)).status  # type: ignore[union-attr]
            return first, second, status

    first, second, status = asyncio.run(run())
    assert first is not None
    assert second is None
    assert status == ANALYSIS_STATUS_PROCESSING


# --- local mock E2E --------------------------------------------------------


def test_mock_end_to_end_flow(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """completed transcript -> POST /analyze -> worker (mock provider) -> GET /analysis."""
    test_client, database, test_settings = client
    seed_completed_meeting(database)
    monkeypatch.setattr(
        analysis_pipeline,
        "build_provider",
        lambda _settings: MockProvider(content=json.dumps(MOCK_PAYLOAD)),
    )

    assert test_client.post("/api/v1/meetings/m1/analyze").status_code == 202

    from app.worker import run_once

    asyncio.run(run_once(database, test_settings))

    body = test_client.get("/api/v1/meetings/m1/analysis").json()
    assert body["status"] == ANALYSIS_STATUS_COMPLETED
    assert body["summary"] == "Özet."
    assert body["topics"] == ["konu"]
    assert body["provider"] == "mock"
    # Ordinal 1 -> persisted start_seconds 5.42; ordinal 2 -> 9.81; ordinal 0 -> 0.0
    assert body["decisions"][0]["timestamp_seconds"] == 5.42
    assert body["action_items"][0]["timestamp_seconds"] == 9.81
    assert body["important_moments"][0]["timestamp_seconds"] == 0.0

    # Transcript is unchanged by the analysis.
    transcript = test_client.get("/api/v1/meetings/m1/transcript").json()
    assert len(transcript["turns"]) == 3
