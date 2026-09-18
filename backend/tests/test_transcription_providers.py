"""Provider architecture tests: selection, ElevenLabs request/response, no fallback.

No test performs a network call: httpx.post is replaced with a fake transport.
"""

from __future__ import annotations

import asyncio
import json
import logging
import wave
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app.config import Settings
from app.db import Database
from app.models import (
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
    Meeting,
    TranscriptTurn,
)
from app.services import pipeline as pipeline_module
from app.services.providers.elevenlabs import ElevenLabsProvider
from app.services.transcription import (
    NormalizedWord,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderUnavailableError,
    build_provider,
    form_turns,
)


def settings(**overrides) -> Settings:
    values = {
        "transcription_provider": "local",
        "llm_provider": "openai_compatible",
        "llm_base_url": None,
        "llm_api_key": None,
        "llm_model": None,
    }
    values.update(overrides)
    return Settings(**values)


def wav_file(tmp_path: Path, seconds: float = 2.0) -> Path:
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def sample_payload() -> dict:
    return {
        "language_code": "tur",
        "text": "Merhaba nasılsın",
        "words": [
            {
                "type": "word",
                "text": "Merhaba",
                "start": 0.1,
                "end": 0.5,
                "speaker_id": "speaker_b",
            },
            {"type": "spacing", "text": " ", "start": 0.5, "end": 0.5},
            {
                "type": "word",
                "text": "nasılsın",
                "start": 0.6,
                "end": 1.2,
                "speaker_id": "speaker_a",
            },
            {"type": "audio_event", "text": "(gülüşme)", "start": 1.2, "end": 1.3},
            {"type": "word", "text": "bilinmeyen", "start": 1.4, "end": 1.6},
        ],
    }


# --- provider selection ----------------------------------------------------


def test_local_is_the_default_provider() -> None:
    provider = build_provider(settings())
    assert provider.name == "local"
    assert settings().transcription_provider == "local"


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ProviderConfigurationError):
        build_provider(settings(transcription_provider="mystery"))


def test_elevenlabs_provider_requires_a_key() -> None:
    provider = ElevenLabsProvider(settings(transcription_provider="elevenlabs"))
    with pytest.raises(ProviderConfigurationError):
        provider.transcribe(
            Path("/nonexistent.wav"),
            requested_speaker_count=None,
            settings=settings(elevenlabs_api_key=""),
        )


# --- ElevenLabs request construction ---------------------------------------


def test_request_fields_are_exact() -> None:
    data = ElevenLabsProvider.build_form_data(settings(), requested_speaker_count=None)
    assert data == {
        "model_id": "scribe_v2",
        "timestamps_granularity": "word",
        "diarize": "true",
        "tag_audio_events": "false",
        "no_verbatim": "false",
        "language_code": "tur",
    }


def test_language_code_is_configurable() -> None:
    data = ElevenLabsProvider.build_form_data(
        settings(elevenlabs_language_code="eng"), requested_speaker_count=None
    )
    assert data["language_code"] == "eng"


def test_known_speaker_count_is_forwarded() -> None:
    data = ElevenLabsProvider.build_form_data(settings(), requested_speaker_count=4)
    assert data["num_speakers"] == "4"


def test_automatic_mode_omits_num_speakers() -> None:
    data = ElevenLabsProvider.build_form_data(settings(), requested_speaker_count=None)
    assert "num_speakers" not in data


def test_threshold_is_omitted_by_default() -> None:
    data = ElevenLabsProvider.build_form_data(settings(), requested_speaker_count=None)
    assert "diarization_threshold" not in data


def test_threshold_is_sent_only_without_known_speaker_count() -> None:
    automatic = ElevenLabsProvider.build_form_data(
        settings(elevenlabs_diarization_threshold=0.35), requested_speaker_count=None
    )
    known = ElevenLabsProvider.build_form_data(
        settings(elevenlabs_diarization_threshold=0.35), requested_speaker_count=3
    )
    assert automatic["diarization_threshold"] == "0.35"
    assert "diarization_threshold" not in known
    assert known["num_speakers"] == "3"


# --- response normalization ------------------------------------------------


def test_parse_words_keeps_only_word_entries() -> None:
    words = ElevenLabsProvider.parse_words(sample_payload())
    assert [word.text for word in words] == ["Merhaba", "nasılsın", "bilinmeyen"]
    assert words[-1].speaker_id is None


def test_missing_speaker_stays_none_and_breaks_turns() -> None:
    words = [
        NormalizedWord(0.0, 0.3, "a", "speaker_b"),
        NormalizedWord(0.4, 0.6, "b", None),
        NormalizedWord(0.7, 1.0, "c", "speaker_b"),
    ]
    turns = form_turns(words)
    assert [turn.speaker for turn in turns] == ["Kişi 1", "Bilinmeyen", "Kişi 1"]


def test_speaker_ids_map_by_first_appearance() -> None:
    words = [
        NormalizedWord(0.0, 0.3, "a", "speaker_b"),
        NormalizedWord(0.4, 0.6, "b", "speaker_a"),
    ]
    turns = form_turns(words)
    assert [turn.speaker for turn in turns] == ["Kişi 1", "Kişi 2"]
    assert "speaker_b" not in json.dumps([turn.speaker for turn in turns])


def test_turn_grouping_merges_consecutive_same_speaker() -> None:
    words = [
        NormalizedWord(0.0, 0.3, "a", "s0"),
        NormalizedWord(0.3, 0.6, "b", "s0"),
        NormalizedWord(0.6, 0.9, "c", "s1"),
    ]
    turns = form_turns(words)
    assert [(turn.speaker, turn.words) for turn in turns] == [("Kişi 1", 2), ("Kişi 2", 1)]


def test_malformed_words_list_is_rejected() -> None:
    with pytest.raises(ProviderResponseError):
        ElevenLabsProvider.parse_words({"words": "nope"})
    with pytest.raises(ProviderResponseError):
        ElevenLabsProvider.parse_words({})


@pytest.mark.parametrize(
    "words",
    [
        [NormalizedWord(-1.0, 0.5, "a", "s0")],
        [NormalizedWord(0.6, 0.4, "a", "s0")],
        [NormalizedWord(1.0, 1.5, "a", "s0"), NormalizedWord(1.2, 1.6, "b", "s0")],
        [NormalizedWord(0.0, 99.0, "a", "s0")],
    ],
)
def test_invalid_timestamps_are_rejected(words: list[NormalizedWord]) -> None:
    with pytest.raises(ProviderResponseError):
        ElevenLabsProvider.validate_timestamps(words, audio_seconds=10.0)


# --- HTTP behavior (fake transport, no network) ----------------------------


class FakeResponse:
    def __init__(
        self, status_code: int, payload: dict | None = None, *, bad_json: bool = False
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self._bad_json = bad_json

    def json(self) -> dict:
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


def call_provider(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeResponse | Exception,
    *,
    tmp_path: Path,
    **overrides,
) -> tuple[object, dict]:
    captured: dict = {}

    def fake_post(url, headers=None, data=None, files=None, timeout=None):
        captured.update(
            {
                "url": url,
                "headers": headers,
                "data": data,
                "timeout": timeout,
                "files": list((files or {}).keys()),
            }
        )
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("httpx.post", fake_post)
    # A dummy key so the request-construction path is reached; real keys never appear.
    overrides.setdefault("elevenlabs_api_key", "sk_test_key_value")
    provider = ElevenLabsProvider(settings(transcription_provider="elevenlabs"))
    outcome = None
    error = None
    try:
        outcome = provider.transcribe(
            wav_file(tmp_path),
            requested_speaker_count=overrides.get("requested_speaker_count"),
            settings=settings(
                transcription_provider="elevenlabs",
                **{k: v for k, v in overrides.items() if k != "requested_speaker_count"},
            ),
        )
    except Exception as exc:  # noqa: BLE001 - the test asserts on the error type
        error = exc
    return (outcome, error), captured


def test_successful_call_normalizes_words(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (outcome, error), captured = call_provider(
        monkeypatch, FakeResponse(200, sample_payload()), tmp_path=tmp_path
    )

    assert error is None
    assert outcome.provider == "elevenlabs"
    assert outcome.model == "scribe_v2"
    assert outcome.language == "tur"
    assert [word.text for word in outcome.words] == ["Merhaba", "nasılsın", "bilinmeyen"]
    assert captured["url"].endswith("/v1/speech-to-text")
    assert captured["files"] == ["file"]
    assert set(captured["headers"]) == {"xi-api-key"}
    assert captured["data"]["model_id"] == "scribe_v2"


def test_api_key_is_only_in_the_header_and_never_logged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sk_test_key_value"
    captured_logs: list[str] = []

    with caplog.at_level(logging.DEBUG):
        (outcome, error), captured = call_provider(
            monkeypatch,
            FakeResponse(200, sample_payload()),
            tmp_path=tmp_path,
            elevenlabs_api_key=secret,
        )
    captured_logs = [record.getMessage() for record in caplog.records]

    assert error is None
    assert captured["headers"]["xi-api-key"] == secret
    assert all(secret not in message for message in captured_logs)
    assert secret not in json.dumps(outcome.words, default=str)


def test_auth_error_fails_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (_, error), _ = call_provider(monkeypatch, FakeResponse(401), tmp_path=tmp_path)
    assert isinstance(error, ProviderRequestError)
    assert "401" in str(error)


def test_quota_error_fails_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (_, error), _ = call_provider(monkeypatch, FakeResponse(429), tmp_path=tmp_path)
    assert isinstance(error, ProviderRequestError)
    assert "quota" in str(error).lower()


def test_server_error_fails_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (_, error), _ = call_provider(monkeypatch, FakeResponse(503), tmp_path=tmp_path)
    assert isinstance(error, ProviderUnavailableError)


def test_timeout_fails_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (_, error), _ = call_provider(
        monkeypatch, httpx.TimeoutException("timeout"), tmp_path=tmp_path
    )
    assert isinstance(error, ProviderUnavailableError)


def test_malformed_json_fails_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (_, error), _ = call_provider(
        monkeypatch, FakeResponse(200, bad_json=True), tmp_path=tmp_path
    )
    assert isinstance(error, ProviderResponseError)


def test_empty_words_fail_typed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (_, error), _ = call_provider(
        monkeypatch, FakeResponse(200, {"words": []}), tmp_path=tmp_path
    )
    assert isinstance(error, ProviderResponseError)


# --- pipeline integration (no fallback, metadata persisted) ----------------


def test_failed_elevenlabs_job_does_not_fall_back_to_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_obj = Settings(
        data_dir=tmp_path / "meetings",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'provider.sqlite'}",
        transcription_provider="elevenlabs",
        elevenlabs_api_key="sk_test_key_value",
    )
    database = Database(settings_obj.database_url)
    asyncio.run(database.init())

    recording_dir = settings_obj.meetings_dir / "m-provider"
    recording_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(recording_dir / "processing.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    local_calls: list[str] = []
    monkeypatch.setattr(
        "app.services.providers.local.transcribe",
        lambda *args, **kwargs: local_calls.append("local-called"),
    )
    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: FakeResponse(401))

    async def run() -> tuple[str, str | None, int, Meeting]:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="m-provider",
                    status="queued",
                    audio_mp3_path="m-provider/meeting.mp3",
                    processing_wav_path="m-provider/processing.wav",
                )
            )
            await session.commit()
            meeting = await pipeline_module.claim_next_meeting(session)
            await pipeline_module.process_meeting(session, meeting, settings_obj)
            refreshed = await session.get(Meeting, "m-provider")
            turns = (await session.execute(select(TranscriptTurn))).scalars().all()
            return refreshed.status, refreshed.processing_error, len(list(turns)), refreshed

    status, error, turn_count, meeting = asyncio.run(run())

    assert status == MEETING_STATUS_FAILED
    assert "401" in (error or "")
    assert turn_count == 0
    assert local_calls == []  # no silent fallback to the local pipeline
    assert meeting.transcription_provider is None


def test_provider_and_model_are_persisted_without_raw_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_obj = Settings(
        data_dir=tmp_path / "meetings",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'provider-ok.sqlite'}",
        transcription_provider="elevenlabs",
        elevenlabs_api_key="sk_test_key_value",
    )
    database = Database(settings_obj.database_url)
    asyncio.run(database.init())

    recording_dir = settings_obj.meetings_dir / "m-ok"
    recording_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(recording_dir / "processing.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: FakeResponse(200, sample_payload()))

    async def run() -> tuple[Meeting, list[TranscriptTurn]]:
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id="m-ok",
                    status="queued",
                    audio_mp3_path="m-ok/meeting.mp3",
                    processing_wav_path="m-ok/processing.wav",
                )
            )
            await session.commit()
            meeting = await pipeline_module.claim_next_meeting(session)
            await pipeline_module.process_meeting(session, meeting, settings_obj)
            turns = list(
                (
                    await session.execute(
                        select(TranscriptTurn).order_by(TranscriptTurn.ordinal)
                    )
                )
                .scalars()
                .all()
            )
            return await session.get(Meeting, "m-ok"), turns

    meeting, turns = asyncio.run(run())

    assert meeting.status == MEETING_STATUS_COMPLETED
    assert meeting.transcription_provider == "elevenlabs"
    assert meeting.transcription_model == "scribe_v2"
    assert [turn.speaker for turn in turns] == ["Kişi 1", "Kişi 2", "Bilinmeyen"]
    # No raw provider payload is persisted anywhere on the meeting row.
    values = json.dumps(meeting.__dict__, default=str)
    assert "language_probability" not in values
    assert "audio_event" not in values
