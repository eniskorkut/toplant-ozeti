"""API tests for grounded meeting Q&A (mocked LLM boundary)."""

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
    MEETING_STATUS_UPLOADED,
    Meeting,
    MeetingChatMessage,
    TranscriptTurn,
)
from app.services import meeting_chat
from app.services.llm_provider import (
    LlmConfigurationError,
    LlmProviderError,
    MockProvider,
    ProviderResponse,
)


@pytest.fixture
def chat_client(tmp_path: Path) -> Iterator[tuple[TestClient, Database]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'chat.sqlite'}")
    asyncio.run(database.init())
    settings = Settings(data_dir=tmp_path / "meetings", database_url=database.url)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    yield TestClient(app), database
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def seed_meeting(
    database: Database,
    meeting_id: str,
    *,
    status: str = MEETING_STATUS_COMPLETED,
    turns: list[tuple[str, str]] | None = None,
    chat: list[tuple[str, str]] | None = None,
) -> None:
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
            for ordinal, (speaker, text) in enumerate(turns or []):
                session.add(
                    TranscriptTurn(
                        meeting_id=meeting_id,
                        ordinal=ordinal,
                        speaker=speaker,
                        start_seconds=float(ordinal),
                        end_seconds=float(ordinal) + 1.0,
                        text=text,
                    )
                )
            for role, content in chat or []:
                session.add(
                    MeetingChatMessage(meeting_id=meeting_id, role=role, content=content)
                )
            await session.commit()

    asyncio.run(run())


def stored_messages(database: Database, meeting_id: str) -> list[tuple[str, str]]:
    async def run() -> list[tuple[str, str]]:
        async with database.session_factory() as session:
            return [
                (message.role, message.content)
                for message in await meeting_chat.list_chat_messages(session, meeting_id)
            ]

    return asyncio.run(run())


def test_chat_answers_and_persists_exchange(chat_client, monkeypatch) -> None:
    client, database = chat_client
    seed_meeting(
        database,
        "m1",
        turns=[("Kişi 1", "Cuma günü yayınlıyoruz."), ("Kişi 2", "Tamam.")],
    )
    monkeypatch.setattr(
        meeting_chat,
        "build_provider",
        lambda _settings: MockProvider(content="Cuma günü yayın kararı alındı."),
    )

    response = client.post("/api/v1/meetings/m1/chat", json={"question": "Ne karar alındı?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Cuma günü yayın kararı alındı."
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assistant = body["messages"][-1]
    assert assistant["content"] == "Cuma günü yayın kararı alındı."
    assert assistant["provider"] == "mock"
    assert assistant["model"] == "mock-model"

    assert stored_messages(database, "m1") == [
        ("user", "Ne karar alındı?"),
        ("assistant", "Cuma günü yayın kararı alındı."),
    ]


def test_chat_history_survives_and_is_returned(chat_client) -> None:
    client, database = chat_client
    seed_meeting(
        database,
        "m2",
        turns=[("Kişi 1", "Merhaba.")],
        chat=[("user", "İlk soru"), ("assistant", "İlk cevap")],
    )

    response = client.get("/api/v1/meetings/m2/chat")

    assert response.status_code == 200
    body = response.json()
    assert body["meeting_id"] == "m2"
    assert [(message["role"], message["content"]) for message in body["messages"]] == [
        ("user", "İlk soru"),
        ("assistant", "İlk cevap"),
    ]


def test_chat_forwards_question_and_stored_history(chat_client, monkeypatch) -> None:
    client, database = chat_client
    seed_meeting(
        database,
        "m3",
        turns=[("Kişi 1", "Merhaba.")],
        chat=[("user", "İlk soru"), ("assistant", "İlk cevap")],
    )
    captured: dict[str, str] = {}

    class RecordingProvider:
        def analyze(self, *, system_prompt: str, user_prompt: str, session_id: str):
            captured["user_prompt"] = user_prompt
            return ProviderResponse(
                content="Tamam.", provider="mock", model="mock-model", latency_seconds=0.0
            )

    monkeypatch.setattr(meeting_chat, "build_provider", lambda _settings: RecordingProvider())

    response = client.post("/api/v1/meetings/m3/chat", json={"question": "Peki ya sonra?"})

    assert response.status_code == 200
    prompt = captured["user_prompt"]
    assert "Transcript:" in prompt
    assert "İlk soru" in prompt
    assert "İlk cevap" in prompt
    assert "Peki ya sonra?" in prompt


def test_chat_clear_removes_history(chat_client) -> None:
    client, database = chat_client
    seed_meeting(
        database,
        "m4",
        turns=[("Kişi 1", "Merhaba.")],
        chat=[("user", "İlk soru"), ("assistant", "İlk cevap")],
    )

    assert client.delete("/api/v1/meetings/m4/chat").status_code == 204
    assert stored_messages(database, "m4") == []


def test_chat_failure_persists_nothing(chat_client, monkeypatch) -> None:
    client, database = chat_client
    seed_meeting(database, "m5", turns=[("Kişi 1", "Merhaba.")])

    def fail(_settings):
        raise LlmProviderError("provider exploded")

    monkeypatch.setattr(meeting_chat, "build_provider", fail)

    assert client.post("/api/v1/meetings/m5/chat", json={"question": "?"}).status_code == 502
    assert stored_messages(database, "m5") == []


def test_chat_rejects_unprocessed_meeting(chat_client) -> None:
    client, database = chat_client
    seed_meeting(database, "m6", status=MEETING_STATUS_UPLOADED)

    response = client.post("/api/v1/meetings/m6/chat", json={"question": "Ne konuşuldu?"})

    assert response.status_code == 409


def test_chat_missing_meeting_404(chat_client) -> None:
    client, _ = chat_client
    assert (
        client.post("/api/v1/meetings/nope/chat", json={"question": "?"}).status_code == 404
    )


def test_chat_validates_empty_question(chat_client) -> None:
    client, database = chat_client
    seed_meeting(database, "m7", turns=[("Kişi 1", "Merhaba.")])

    assert (
        client.post("/api/v1/meetings/m7/chat", json={"question": ""}).status_code == 422
    )


def test_chat_returns_503_when_llm_not_configured(chat_client, monkeypatch) -> None:
    client, database = chat_client
    seed_meeting(database, "m8", turns=[("Kişi 1", "Merhaba.")])

    def raise_config(_settings):
        raise LlmConfigurationError("not configured")

    monkeypatch.setattr(meeting_chat, "build_provider", raise_config)

    response = client.post("/api/v1/meetings/m8/chat", json={"question": "Ne konuşuldu?"})

    assert response.status_code == 503
