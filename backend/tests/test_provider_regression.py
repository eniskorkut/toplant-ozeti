"""Regression tests for the provider refactor: local behavior, migration, settings."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from sqlalchemy import text

from app.config import Settings
from app.db import Database
from app.services.diarization import DiarizationResult, DiarizationSegment
from app.services.merge import (
    DiarizationSegment as MergeSegment,
)
from app.services.merge import (
    Word as MergeWord,
)
from app.services.merge import (
    merge_words,
)
from app.services.providers.local import LocalProvider
from app.services.stt import SttResult, SttWord
from app.services.transcription import form_turns

STT_WORDS = [
    SttWord(0.0, 0.4, "Merhaba"),
    SttWord(0.5, 0.9, "dunya"),
    SttWord(1.2, 1.6, "nasilsin"),
    SttWord(5.0, 5.4, "kayip"),
    SttWord(6.0, 6.4, "tekrar"),
]
DIARIZATION = [
    DiarizationSegment(0.0, 1.7, "cluster_5"),
    DiarizationSegment(5.9, 7.0, "cluster_2"),
]


def test_local_provider_matches_the_production_merge_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The refactored local path must produce byte-identical turns to merge_words."""
    settings = Settings(
        llm_provider="openai_compatible", llm_base_url=None, llm_api_key=None, llm_model=None
    )

    from app.services.providers import local as local_module

    monkeypatch.setattr(
        local_module,
        "transcribe",
        lambda path, settings: SttResult(
            words=STT_WORDS,
            segments=[],
            text="Merhaba dunya nasilsin kayip tekrar",
            language="tr",
            inference_seconds=1.0,
        ),
    )
    monkeypatch.setattr(
        local_module,
        "diarize",
        lambda path, settings, requested_speaker_count=None: DiarizationResult(
            segments=DIARIZATION,
            num_speakers=2,
            inference_seconds=0.5,
            model_load_seconds=0.1,
        ),
    )

    result = LocalProvider().transcribe(
        tmp_path / "audio.wav", requested_speaker_count=None, settings=settings
    )
    new_turns = [
        (turn.speaker, round(turn.start, 3), round(turn.end, 3), turn.text)
        for turn in form_turns(result.words)
    ]

    reference = merge_words(
        [MergeWord(word.start, word.end, word.text) for word in STT_WORDS],
        [MergeSegment(segment.start, segment.end, segment.speaker) for segment in DIARIZATION],
        tolerance=settings.merge_boundary_tolerance_seconds,
    )
    old_turns = [
        (turn.speaker, round(turn.start, 3), round(turn.end, 3), turn.text)
        for turn in reference.turns
    ]

    assert new_turns == old_turns
    assert result.provider == "local"
    assert result.model == "whisper-large-v3-turbo-q8"
    assert result.language == "tr"
    assert result.latency_seconds == 1.5


def test_elevenlabs_settings_use_the_documented_env_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_env_value")
    monkeypatch.setenv("ELEVENLABS_STT_MODEL", "scribe_v2_test")
    monkeypatch.setenv("ELEVENLABS_LANGUAGE_CODE", "eng")
    monkeypatch.setenv("ELEVENLABS_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("ELEVENLABS_DIARIZATION_THRESHOLD", "0.4")
    monkeypatch.setenv("MEETING_TRANSCRIPTION_PROVIDER", "elevenlabs")

    settings = Settings()

    assert settings.transcription_provider == "elevenlabs"
    assert settings.elevenlabs_api_key is not None
    assert settings.elevenlabs_api_key.get_secret_value() == "sk_env_value"
    assert settings.elevenlabs_stt_model == "scribe_v2_test"
    assert settings.elevenlabs_language_code == "eng"
    assert settings.elevenlabs_timeout_seconds == 42
    assert settings.elevenlabs_diarization_threshold == 0.4
    # SecretStr keeps the value out of reprs/log lines.
    assert "sk_env_value" not in repr(settings)


def test_empty_env_values_are_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compose passes empty strings for unset variables; they must not crash settings."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "")
    monkeypatch.setenv("ELEVENLABS_DIARIZATION_THRESHOLD", "")
    monkeypatch.setenv("MEETING_TRANSCRIPTION_PROVIDER", "local")

    settings = Settings()

    assert settings.elevenlabs_api_key is None
    assert settings.elevenlabs_diarization_threshold is None
    assert settings.transcription_provider == "local"


def test_additive_migration_upgrades_an_existing_database(tmp_path: Path) -> None:
    """create_all does not alter tables; init() must add the provider columns."""
    url = f"sqlite+aiosqlite:///{tmp_path / 'legacy.sqlite'}"
    database = Database(url)

    async def create_legacy_schema() -> None:
        async with database.engine.begin() as connection:
            await connection.exec_driver_sql(
                """
                CREATE TABLE meetings (
                    id VARCHAR(32) PRIMARY KEY,
                    created_at DATETIME,
                    duration_seconds FLOAT,
                    status VARCHAR(16),
                    audio_mp3_path VARCHAR(255),
                    processing_wav_path VARCHAR(255),
                    requested_speaker_count INTEGER,
                    processing_error TEXT
                )
                """
            )

    async def columns() -> set[str]:
        async with database.engine.begin() as connection:
            result = await connection.exec_driver_sql("PRAGMA table_info(meetings)")
            return {row[1] for row in result.all()}

    asyncio.run(create_legacy_schema())
    before = asyncio.run(columns())
    assert "transcription_provider" not in before

    asyncio.run(database.init())
    after = asyncio.run(columns())

    assert {"transcription_provider", "transcription_model"} <= after
    # idempotent: running init again keeps a single column set
    asyncio.run(database.init())
    assert after == asyncio.run(columns())

    async def insert_legacy_row() -> None:
        async with database.engine.begin() as connection:
            await connection.execute(
                text(
                    "INSERT INTO meetings (id, status, audio_mp3_path, processing_wav_path, "
                    "transcription_provider) VALUES "
                    "('legacy', 'completed', 'a.mp3', 'a.wav', 'local')"
                )
            )

    asyncio.run(insert_legacy_row())
