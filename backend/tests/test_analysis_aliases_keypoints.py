"""Alias-aware analysis prompts, grounded key points, refresh and legacy rows."""

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
    ANALYSIS_STATUS_PROCESSING,
    ANALYSIS_STATUS_QUEUED,
    MEETING_STATUS_COMPLETED,
    Meeting,
    MeetingAnalysis,
    MeetingSpeakerAlias,
    TranscriptTurn,
)
from app.services import analysis_pipeline
from app.services.analysis_pipeline import payload_from_row
from app.services.analysis_prompt import serialize_transcript
from app.services.analysis_schema import TranscriptTurnView, validate_payload
from app.services.llm_provider import ProviderResponse

PAYLOAD = {
    "summary": "Ahmet yayın planını anlattı, Mehmet testleri üstlendi.",
    "key_points": [
        {"text": "Yayın hedefi cuma günü olarak belirlendi.", "source_turn_ordinals": [0]},
        {"text": "Testler perşembe tamamlanacak.", "source_turn_ordinals": [1]},
    ],
    "topics": ["yayın"],
    "decisions": [{"text": "Cuma yayınlanacak.", "source_turn_ordinals": [0]}],
    "action_items": [
        {
            "task": "Testleri tamamla",
            "owner": "Kişi 2",
            "due_date_text": "perşembe",
            "source_turn_ordinals": [1],
        }
    ],
    "important_moments": [
        {"title": "Yayın kararı", "description": "Cuma kararı verildi.", "source_turn_ordinal": 0}
    ],
}


class CapturingProvider:
    name = "capture"
    model = "capture-model"

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or PAYLOAD
        self.prompts: list[str] = []

    def analyze(self, *, system_prompt: str, user_prompt: str, session_id: str) -> ProviderResponse:
        self.prompts.append(user_prompt)
        return ProviderResponse(
            content=json.dumps(self.payload, ensure_ascii=False),
            provider=self.name,
            model=self.model,
            latency_seconds=0.1,
        )


def settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "data_dir": tmp_path / "meetings",
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'aliases.sqlite'}",
        "llm_provider": "mock",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def context(tmp_path: Path) -> Iterator[tuple[TestClient, Database, Settings]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'analysis.sqlite'}")
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


def seed_meeting(
    database: Database, meeting_id: str, *, aliases: dict[str, str] | None = None
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
            session.add_all(
                [
                    TranscriptTurn(
                        meeting_id=meeting_id,
                        ordinal=0,
                        speaker="Kişi 1",
                        start_seconds=0.0,
                        end_seconds=5.0,
                        text="Backend değişikliklerini ben tamamlayacağım.",
                    ),
                    TranscriptTurn(
                        meeting_id=meeting_id,
                        ordinal=1,
                        speaker="Kişi 2",
                        start_seconds=5.42,
                        end_seconds=8.0,
                        text="Testleri ben yapacağım.",
                    ),
                ]
            )
            for canonical, display in (aliases or {}).items():
                session.add(
                    MeetingSpeakerAlias(
                        meeting_id=meeting_id,
                        canonical_speaker=canonical,
                        display_name=display,
                    )
                )
            await session.commit()

    asyncio.run(run())


async def run_pipeline(database: Database, meeting_id: str, provider: CapturingProvider) -> None:
    from app.services.analysis_pipeline import process_analysis

    async with database.session_factory() as session:
        analysis = (
            await session.execute(
                select(MeetingAnalysis).where(MeetingAnalysis.meeting_id == meeting_id)
            )
        ).scalar_one_or_none()
        if analysis is None:
            analysis = MeetingAnalysis(meeting_id=meeting_id, status=ANALYSIS_STATUS_QUEUED)
            session.add(analysis)
            await session.commit()
            await session.refresh(analysis)
        await process_analysis(session, analysis, settings(Path("/tmp"), llm_provider="mock"))


# --- prompt serialization ----------------------------------------------------


def test_serialize_transcript_includes_alias_in_parentheses() -> None:
    turns = [
        TranscriptTurnView(0, "Kişi 1", 0.0, "Merhaba", alias="Ahmet"),
        TranscriptTurnView(1, "Kişi 2", 5.42, "Tamam"),
    ]
    text = serialize_transcript(turns)
    assert "Kişi 1 (Ahmet): Merhaba" in text
    assert "[1] 00:05.420 Kişi 2: Tamam" in text  # no alias -> canonical only


def test_prompt_uses_current_meeting_aliases_only(context, monkeypatch) -> None:
    client, database, _ = context
    seed_meeting(database, "m1", aliases={"Kişi 1": "Ahmet", "Kişi 2": "Mehmet"})
    seed_meeting(database, "m2", aliases={"Kişi 1": "Zeynep"})
    seed_meeting(database, "m3")

    provider = CapturingProvider()
    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: provider)
    asyncio.run(run_pipeline(database, "m1", provider))

    prompt = provider.prompts[0]
    assert "Kişi 1 (Ahmet)" in prompt
    assert "Kişi 2 (Mehmet)" in prompt
    assert "Zeynep" not in prompt  # another meeting's alias never leaks


def test_owner_must_stay_canonical_even_with_alias() -> None:
    turns = [TranscriptTurnView(0, "Kişi 1", 0.0, "Merhaba", alias="Ahmet")]
    raw = json.dumps(
        {
            "summary": "Özet.",
            "key_points": [],
            "topics": [],
            "decisions": [],
            "action_items": [
                {"task": "İş", "owner": "Ahmet", "due_date_text": None, "source_turn_ordinals": [0]}
            ],
            "important_moments": [],
        }
    )
    payload, errors = validate_payload(raw, turns)
    assert payload is None
    assert any("unknown owner" in error for error in errors)


# --- key points --------------------------------------------------------------


def test_key_point_requires_valid_ordinals() -> None:
    turns = [TranscriptTurnView(0, "Kişi 1", 0.0, "Merhaba")]
    base = {
        "summary": "Özet.",
        "key_points": [{"text": "Fikir", "source_turn_ordinals": []}],
        "topics": [],
        "decisions": [],
        "action_items": [],
        "important_moments": [],
    }
    payload, errors = validate_payload(json.dumps(base), turns)
    assert payload is None
    assert any("key_points" in error for error in errors)

    base["key_points"] = [{"text": "Fikir", "source_turn_ordinals": [7]}]
    payload, errors = validate_payload(json.dumps(base), turns)
    assert payload is None
    assert any("does not exist" in error for error in errors)


def test_pipeline_stores_key_points_and_api_derives_timestamps(context, monkeypatch) -> None:
    client, database, _ = context
    seed_meeting(database, "m1")
    provider = CapturingProvider()
    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: provider)
    asyncio.run(run_pipeline(database, "m1", provider))

    body = client.get("/api/v1/meetings/m1/analysis").json()
    assert body["status"] == "completed"
    assert [point["text"] for point in body["key_points"]] == [
        "Yayın hedefi cuma günü olarak belirlendi.",
        "Testler perşembe tamamlanacak.",
    ]
    # Timestamps come from the FIRST referenced ordinal (never from the model).
    assert body["key_points"][0]["timestamp_seconds"] == 0.0
    assert body["key_points"][1]["timestamp_seconds"] == 5.42
    assert body["action_items"][0]["owner"] == "Kişi 2"  # canonical stays canonical


def test_legacy_analysis_without_key_points_loads_empty(context) -> None:
    client, database, _ = context
    seed_meeting(database, "m1")

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                MeetingAnalysis(
                    meeting_id="m1",
                    status=ANALYSIS_STATUS_COMPLETED,
                    summary="Eski özet.",
                    topics_json="[]",
                    decisions_json="[]",
                    action_items_json="[]",
                    important_moments_json="[]",
                    key_points_json=None,
                )
            )
            await session.commit()

    asyncio.run(seed())
    body = client.get("/api/v1/meetings/m1/analysis").json()
    assert body["key_points"] == []
    assert body["summary"] == "Eski özet."

    async def load_payload():
        async with database.session_factory() as session:
            row = (
                await session.execute(
                    select(MeetingAnalysis).where(MeetingAnalysis.meeting_id == "m1")
                )
            ).scalar_one()
            return payload_from_row(row)

    payload = asyncio.run(load_payload())
    assert payload is not None and payload.key_points == []


# --- refresh -----------------------------------------------------------------


def test_refresh_rebuilds_with_current_aliases(context, monkeypatch) -> None:
    client, database, _ = context
    seed_meeting(database, "m1")
    provider = CapturingProvider()
    monkeypatch.setattr(analysis_pipeline, "build_provider", lambda _settings: provider)
    asyncio.run(run_pipeline(database, "m1", provider))
    assert "Kişi 1" in provider.prompts[0]

    # Rename, then explicitly refresh.
    assert (
        client.put(
            "/api/v1/meetings/m1/speakers/Kişi 1/alias", json={"display_name": "Ahmet"}
        ).status_code
        == 200
    )
    refreshed = client.post("/api/v1/meetings/m1/analyze?refresh=true")
    assert refreshed.status_code == 202
    assert refreshed.json()["status"] == "queued"

    asyncio.run(run_pipeline(database, "m1", provider))
    assert "Kişi 1 (Ahmet)" in provider.prompts[1]


def test_refresh_conflicts_while_analysis_running(context) -> None:
    client, database, _ = context
    seed_meeting(database, "m1")

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(MeetingAnalysis(meeting_id="m1", status=ANALYSIS_STATUS_PROCESSING))
            await session.commit()

    asyncio.run(seed())
    response = client.post("/api/v1/meetings/m1/analyze?refresh=true")
    assert response.status_code == 409


def test_plain_analyze_stays_idempotent_when_completed(context) -> None:
    client, database, _ = context
    seed_meeting(database, "m1")

    async def seed() -> None:
        async with database.session_factory() as session:
            session.add(
                MeetingAnalysis(
                    meeting_id="m1", status=ANALYSIS_STATUS_COMPLETED, summary="Bitti."
                )
            )
            await session.commit()

    asyncio.run(seed())
    response = client.post("/api/v1/meetings/m1/analyze")
    assert response.status_code == 202
    assert response.json()["status"] == "completed"
    assert response.json()["summary"] == "Bitti."


# --- migration ---------------------------------------------------------------


def test_additive_migration_is_idempotent(tmp_path: Path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'migrate.sqlite'}")

    async def init_twice() -> set[str]:
        await database.init()
        await database.init()
        async with database.engine.begin() as connection:
            rows = await connection.exec_driver_sql(
                "PRAGMA table_info(meeting_analyses)"
            )
            return {row[1] for row in rows.all()}

    try:
        columns = asyncio.run(init_twice())
    finally:
        asyncio.run(database.dispose())
    assert "key_points_json" in columns
