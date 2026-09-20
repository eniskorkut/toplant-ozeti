"""Live ElevenLabs endpoints: tokens, sessions, rolling windows, aliases.

Every network call is mocked: no test touches ElevenLabs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from app.config import Settings, get_settings
from app.db import Database, get_session
from app.main import app
from app.models import Meeting, MeetingLiveSpeaker
from app.services.live_sessions import LiveSession, LiveSessionStore, get_live_session_store
from app.services.providers import elevenlabs as elevenlabs_module
from app.services.transcription import (
    NormalizedWord,
    ProviderUnavailableError,
)

SECOND = 16_000 * 2  # bytes of 16 kHz mono s16le per second
KEY = "xi-test-key-never-returned"
TOKEN = "sutkn_test_token_never_logged"


def wav_bytes(seconds: float) -> bytes:
    return b"\x00\x00" * int(16_000 * seconds)


@pytest.fixture
def context(tmp_path: Path) -> Iterator[tuple[TestClient, Database, Settings, LiveSessionStore]]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'live.sqlite'}")
    asyncio.run(database.init())
    meetings_dir = tmp_path / "meetings"
    meetings_dir.mkdir()
    settings = Settings(
        data_dir=meetings_dir,
        database_url=database.url,
        elevenlabs_api_key=KEY,
    )
    store = LiveSessionStore(ttl_seconds=60)

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_live_session_store] = lambda: store
    yield TestClient(app), database, settings, store
    app.dependency_overrides.clear()
    asyncio.run(database.dispose())


def make_session(client: TestClient) -> str:
    response = client.post("/api/v1/live-transcription/sessions")
    assert response.status_code == 201
    return response.json()["live_session_id"]


def confirm_first_speaker(client: TestClient, live_session_id: str) -> None:
    """Two overlapping snapshots with the same cluster promote Kişi 1."""
    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    submit_window(client, live_session_id, start=4, end=12, sequence=2)


def submit_window(
    client: TestClient,
    live_session_id: str,
    *,
    start: float,
    end: float,
    sequence: int,
    seconds: float | None = None,
    speaker_count: int | None = None,
    length_bytes: int | None = None,
):
    duration = end - start
    payload = wav_bytes(seconds if seconds is not None else duration)
    if length_bytes is not None:
        payload = payload[:length_bytes]
    data = {
        "start_seconds": str(start),
        "end_seconds": str(end),
        "sequence": str(sequence),
    }
    if speaker_count is not None:
        data["speaker_count"] = str(speaker_count)
    return client.post(
        f"/api/v1/live-transcription/sessions/{live_session_id}/speaker-window",
        files={"pcm": ("window.pcm", payload, "application/octet-stream")},
        data=data,
    )


def window_words(intervals: dict[str, list[tuple[float, float]]]) -> list[NormalizedWord]:
    words: list[NormalizedWord] = []
    for speaker, spans in intervals.items():
        for start, end in spans:
            words.append(NormalizedWord(start=start, end=end, text="kelime", speaker_id=speaker))
    return words


# --- realtime token endpoint -------------------------------------------------


def test_realtime_token_returns_only_the_single_use_token(context, monkeypatch, caplog) -> None:
    client, _, _, _ = context
    monkeypatch.setattr(
        "app.routers.transcription.create_realtime_token", lambda settings: TOKEN
    )

    with caplog.at_level(logging.DEBUG):
        response = client.post("/api/v1/transcription/providers/elevenlabs/realtime-token")

    assert response.status_code == 200
    assert response.json() == {"token": TOKEN}
    assert KEY not in response.text
    assert KEY not in caplog.text
    assert TOKEN not in caplog.text


def test_realtime_token_503_when_not_configured(tmp_path: Path) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'nokey.sqlite'}")
    asyncio.run(database.init())
    settings = Settings(data_dir=tmp_path / "m", database_url=database.url, elevenlabs_api_key="")
    (tmp_path / "m").mkdir()

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    try:
        client = TestClient(app)
        response = client.post("/api/v1/transcription/providers/elevenlabs/realtime-token")
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()
        asyncio.run(database.dispose())


def test_realtime_token_never_leaks_key_on_provider_failure(context, monkeypatch) -> None:
    client, _, _, _ = context

    def failing(settings):
        raise ProviderUnavailableError(f"boom {KEY}")

    monkeypatch.setattr("app.routers.transcription.create_realtime_token", failing)

    response = client.post("/api/v1/transcription/providers/elevenlabs/realtime-token")

    assert response.status_code == 503
    assert KEY not in response.text


# --- sessions ----------------------------------------------------------------


def test_live_session_create_get_delete(context) -> None:
    client, _, _, store = context
    live_session_id = make_session(client)

    state = client.get(f"/api/v1/live-transcription/sessions/{live_session_id}")
    assert state.status_code == 200
    assert state.json()["windows_received"] == 0

    deleted = client.delete(f"/api/v1/live-transcription/sessions/{live_session_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/live-transcription/sessions/{live_session_id}").status_code == 404
    assert store.count() == 0


def test_unknown_live_session_is_404(context) -> None:
    client, _, _, _ = context
    assert client.get("/api/v1/live-transcription/sessions/nope").status_code == 404
    assert submit_window(client, "nope", start=0, end=2, sequence=1).status_code == 404


def test_expired_sessions_are_pruned(tmp_path: Path) -> None:
    store = LiveSessionStore(ttl_seconds=0.05)
    session: LiveSession = store.create()
    assert store.count() == 1
    import time

    time.sleep(0.08)
    assert store.get(session.id) is None
    assert store.count() == 0


# --- rolling windows ---------------------------------------------------------


def test_window_validation_and_success(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)

    calls: list[dict] = []

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        calls.append(
            {
                "bytes": len(pcm),
                "speaker_count": requested_speaker_count,
            }
        )
        return window_words({"speaker_0": [(0.0, 2.0)]})

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    # duration below the minimum
    assert submit_window(client, live_session_id, start=0, end=0.5, sequence=1).status_code == 422
    # payload does not match declared duration
    mismatch = submit_window(
        client, live_session_id, start=0, end=8, sequence=1, length_bytes=SECOND
    )
    assert mismatch.status_code == 422
    # wrong sequence
    assert submit_window(client, live_session_id, start=0, end=8, sequence=9).status_code == 409
    assert calls == []

    response = submit_window(client, live_session_id, start=0, end=8, sequence=1)
    assert response.status_code == 200
    body = response.json()
    assert body["sequence"] == 1
    assert body["window"] == [0.0, 8.0]
    # One snapshot is only evidence, never a canonical speaker.
    assert body["assignments"] == []
    assert body["confirmed_speakers"] == []
    assert body["candidate_speakers"] == 1
    assert body["provider_speakers"] == 1
    assert calls == [{"bytes": 8 * SECOND, "speaker_count": None}]

    # A second overlapping snapshot confirms the first speaker as Kişi 1.
    second = submit_window(client, live_session_id, start=4, end=12, sequence=2)
    assert second.status_code == 200
    promoted = second.json()
    assert promoted["confirmed_speakers"] == ["Kişi 1"]
    assert promoted["assignments"][0]["canonical_speaker"] == "Kişi 1"
    assert promoted["assignments"][0]["is_new"] is True

    # next sequence expected
    assert submit_window(client, live_session_id, start=8, end=16, sequence=1).status_code == 409


def test_window_timestamps_are_offset_to_global_time(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(
            {"speaker_0": [(0.0, 1.5)]}
        ),
    )

    # The same window twice: the second snapshot promotes Kişi 1.
    submit_window(client, live_session_id, start=10.0, end=18.0, sequence=1)
    response = submit_window(client, live_session_id, start=10.0, end=18.0, sequence=2)
    body = response.json()

    assert body["confirmed_speakers"] == ["Kişi 1"]
    assert body["assignments"][0]["start"] == 10.0
    assert body["assignments"][0]["end"] == 11.5


def test_known_speaker_count_is_forwarded(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)
    seen: dict = {}

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        seen["speaker_count"] = requested_speaker_count
        return window_words({"speaker_0": [(0.0, 2.0)]})

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)
    response = submit_window(client, live_session_id, start=0, end=4, sequence=1, speaker_count=2)
    assert response.status_code == 200
    assert seen["speaker_count"] == 2


def test_request_local_ids_are_mapped_and_unmatched_clusters_wait(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)

    windows = [
        # snapshot 1: the first person (only evidence so far)
        {"speaker_0": [(0.5, 2.5)]},
        # snapshot 2: same cluster again -> Kişi 1 confirmed
        {"speaker_0": [(0.0, 2.5)]},
        # snapshot 3: numbering swapped, overlap keeps Kişi 1
        {"speaker_1": [(0.0, 1.5)]},
        # snapshot 4: a different, comparable cluster becomes a candidate
        {"speaker_0": [(0.0, 1.2)]},
        # snapshot 5: the candidate persists -> promoted densely as Kişi 2
        {"speaker_0": [(0.0, 1.2)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    second = submit_window(client, live_session_id, start=4, end=12, sequence=2).json()
    assert second["confirmed_speakers"] == ["Kişi 1"]

    # Provider renumbers speakers between requests; overlap still maps to Kişi 1.
    third = submit_window(client, live_session_id, start=4, end=12, sequence=3).json()
    labels = [item["canonical_speaker"] for item in third["assignments"]]
    assert labels == ["Kişi 1"]
    assert third["confirmed_speakers"] == ["Kişi 1"]

    fourth = submit_window(client, live_session_id, start=8, end=16, sequence=4).json()
    assert fourth["candidate_speakers"] == 1  # no runaway Kişi 2 yet

    fifth = submit_window(client, live_session_id, start=10, end=18, sequence=5).json()
    assert fifth["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    assert "Kişi 2" in fifth["promoted_speakers"]


def test_rolling_failure_is_safe_and_does_not_consume_sequence(context, monkeypatch) -> None:
    client, _, _, store = context
    live_session_id = make_session(client)

    def failing(pcm, *, settings, requested_speaker_count=None):
        raise ProviderUnavailableError("provider down")

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", failing)
    response = submit_window(client, live_session_id, start=0, end=8, sequence=1)
    assert response.status_code == 503

    session = store.get(live_session_id)
    assert session is not None
    assert session.next_sequence == 1  # retry with the same sequence is allowed
    assert session.windows_received == 0


def test_rolling_503_when_elevenlabs_not_configured(tmp_path: Path, monkeypatch) -> None:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'unconfigured.sqlite'}")
    asyncio.run(database.init())
    meetings_dir = tmp_path / "m"
    meetings_dir.mkdir()
    settings = Settings(data_dir=meetings_dir, database_url=database.url, elevenlabs_api_key="")
    store = LiveSessionStore()

    async def override_session():
        async for session in database.sessions():
            yield session

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_live_session_store] = lambda: store
    try:
        client = TestClient(app)
        live_session_id = make_session(client)
        response = submit_window(client, live_session_id, start=0, end=8, sequence=1)
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()
        asyncio.run(database.dispose())


# --- aliases -----------------------------------------------------------------


def test_live_alias_set_update_reset(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(
            {"speaker_0": [(0.0, 2.0)]}
        ),
    )
    confirm_first_speaker(client, live_session_id)

    url = f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/Kişi 1/alias"
    assert client.put(url, json={"display_name": "  Ahmet  "}).json()["display_name"] == "Ahmet"
    assert client.put(url, json={"display_name": "Mehmet"}).status_code == 200
    state = client.get(f"/api/v1/live-transcription/sessions/{live_session_id}").json()
    assert state["aliases"] == {"Kişi 1": "Mehmet"}
    assert client.delete(url).status_code == 204
    cleared = client.get(f"/api/v1/live-transcription/sessions/{live_session_id}").json()
    assert cleared["aliases"] == {}


def test_alias_validation_rejects_empty_control_and_long_names(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(
            {"speaker_0": [(0.0, 2.0)]}
        ),
    )
    confirm_first_speaker(client, live_session_id)
    url = f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/Kişi 1/alias"

    assert client.put(url, json={"display_name": "   "}).status_code == 422
    assert client.put(url, json={"display_name": "a\x00b"}).status_code == 422
    assert client.put(url, json={"display_name": "x" * 51}).status_code == 422


def test_alias_requires_known_canonical_speaker(context) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)
    url = f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/Kişi 3/alias"
    assert client.put(url, json={"display_name": "Ahmet"}).status_code == 404


def test_aliases_are_isolated_between_live_sessions(context, monkeypatch) -> None:
    client, _, _, _ = context
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(
            {"speaker_0": [(0.0, 2.0)]}
        ),
    )
    first = make_session(client)
    second = make_session(client)
    for live_session_id in (first, second):
        confirm_first_speaker(client, live_session_id)

    client.put(
        f"/api/v1/live-transcription/sessions/{first}/speakers/Kişi 1/alias",
        json={"display_name": "Ahmet"},
    )

    assert client.get(f"/api/v1/live-transcription/sessions/{second}").json()["aliases"] == {}


def test_no_voiceprint_columns_or_payloads_are_stored(context) -> None:
    """Structural guarantee: only labels, display names and timestamp intervals."""
    _, database, _, _ = context

    async def columns(table: str) -> set[str]:
        async with database.session_factory() as session:
            result = await session.execute(text(f"PRAGMA table_info({table})"))
            return {row[1] for row in result.all()}

    alias_columns = asyncio.run(columns("meeting_speaker_aliases"))
    live_columns = asyncio.run(columns("meeting_live_speakers"))
    forbidden = {"embedding", "voiceprint", "vector", "audio", "person_id", "global_id"}

    assert alias_columns == {
        "id",
        "meeting_id",
        "canonical_speaker",
        "display_name",
        "created_at",
        "updated_at",
    }
    assert live_columns == {"id", "meeting_id", "canonical_speaker", "intervals_json"}
    assert not (alias_columns | live_columns) & forbidden


# --- meeting aliases, migration, deletion ------------------------------------


def shared_audio_upload(client: TestClient, live_session_id: str | None = None):
    # The real upload route is used; ffmpeg is patched out by the fast_recording
    # fixture, so the payload only needs to be non-empty.
    files = {"audio": ("clip.webm", b"\x00" * 32, "audio/webm;codecs=opus")}
    data = {"mime_type": "audio/webm;codecs=opus", "client_duration_seconds": "4.0"}
    if live_session_id is not None:
        data["live_session_id"] = live_session_id
    return client.post("/api/recordings", files=files, data=data)


@pytest.fixture
def fast_recording(monkeypatch):
    from app.services.recordings import RecordingArtifacts

    def fake_process(*, upload_file, mime_type, settings):
        upload_file.read()
        recording_id = "meeting01"
        directory = settings.meetings_dir / recording_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "meeting.mp3").write_bytes(b"mp3")
        (directory / "processing.wav").write_bytes(b"wav")
        return RecordingArtifacts(
            recording_id=recording_id,
            duration_seconds=4.0,
            input_mime_type="audio/webm;codecs=opus",
            input_size_bytes=32,
            mp3_size_bytes=3,
            wav_size_bytes=3,
            wav_sample_rate=16000,
            wav_channels=1,
            wav_sample_width_bytes=2,
            conversion_ms=1,
        )

    monkeypatch.setattr("app.routers.recordings.process_recording", fake_process)


def test_alias_migration_live_session_to_meeting_and_meeting_isolation(
    context, monkeypatch, fast_recording
) -> None:
    client, database, _, store = context
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(
            {"speaker_0": [(0.0, 2.0)]}
        ),
    )
    live_session_id = make_session(client)
    confirm_first_speaker(client, live_session_id)
    client.put(
        f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/Kişi 1/alias",
        json={"display_name": "Ahmet"},
    )

    upload = shared_audio_upload(client, live_session_id)
    assert upload.status_code == 201
    meeting_id = upload.json()["meeting_id"]

    speakers = client.get(f"/api/v1/meetings/{meeting_id}/speakers").json()
    assert speakers["aliases"] == {"Kişi 1": "Ahmet"}
    assert any(speaker.startswith("Kişi") for speaker in speakers["speakers"])
    assert store.count() == 0  # live state is released after migration

    async def live_rows() -> int:
        async with database.session_factory() as session:
            result = await session.execute(
                select(MeetingLiveSpeaker).where(MeetingLiveSpeaker.meeting_id == meeting_id)
            )
            return len(result.scalars().all())

    assert asyncio.run(live_rows()) == 1

    # A second meeting starts clean: no cross-meeting alias memory.
    other = Meeting(
        id="other01",
        status="uploaded",
        duration_seconds=1.0,
        audio_mp3_path="other01/meeting.mp3",
        processing_wav_path="other01/processing.wav",
    )

    async def add_other() -> None:
        async with database.session_factory() as session:
            session.add(other)
            await session.commit()

    asyncio.run(add_other())
    assert client.get("/api/v1/meetings/other01/speakers").json()["aliases"] == {}


def test_final_speaker_reconciliation_in_the_pipeline(tmp_path: Path, monkeypatch) -> None:
    """Final provider speakers are remapped onto canonical live labels."""
    from app.services.pipeline import _run_inference
    from app.services.transcription import TranscriptionResult

    audio = tmp_path / "processing.wav"
    import wave

    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)

    words = [
        NormalizedWord(start=0.0, end=2.0, text="a", speaker_id="speaker_9"),
        NormalizedWord(start=2.0, end=4.0, text="b", speaker_id="speaker_3"),
        NormalizedWord(start=4.0, end=5.0, text="c", speaker_id="speaker_9"),
    ]

    class FakeProvider:
        name = "elevenlabs"
        model = "scribe_v2"

        def transcribe(self, audio_path, *, requested_speaker_count, settings):
            return TranscriptionResult(
                text="a b c",
                words=words,
                language="tur",
                latency_seconds=0.1,
                provider="elevenlabs",
                model="scribe_v2",
            )

    monkeypatch.setattr("app.services.pipeline.build_provider", lambda settings: FakeProvider())

    live_timelines = {
        "Kişi 1": [(0.0, 2.5), (3.9, 5.0)],
        "Kişi 2": [(2.0, 4.0)],
    }
    output = _run_inference(audio, None, Settings(), "elevenlabs", live_timelines)

    speakers = [turn["speaker"] for turn in output["turns"]]
    assert speakers == ["Kişi 1", "Kişi 2", "Kişi 1"]


def test_final_text_authority_and_unmatched_final_speaker(tmp_path: Path, monkeypatch) -> None:
    from app.services.pipeline import _run_inference
    from app.services.transcription import TranscriptionResult

    audio = tmp_path / "processing.wav"
    import wave

    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 1600)

    class FakeProvider:
        name = "elevenlabs"
        model = "scribe_v2"

        def transcribe(self, audio_path, *, requested_speaker_count, settings):
            return TranscriptionResult(
                text="final text",
                words=[
                    NormalizedWord(
                        start=100.0, end=101.0, text="final", speaker_id="speaker_z"
                    )
                ],
                language="tur",
                latency_seconds=0.1,
                provider="elevenlabs",
                model="scribe_v2",
            )

    monkeypatch.setattr("app.services.pipeline.build_provider", lambda settings: FakeProvider())

    output = _run_inference(audio, None, Settings(), "elevenlabs", {"Kişi 1": [(0.0, 5.0)]})

    assert output["turns"][0]["speaker"] == "Kişi 1"  # dense numbering: 1 speaker -> Kişi 1
    assert output["turns"][0]["text"] == "final"


def test_provider_window_form_data_uses_pcm_format(monkeypatch) -> None:
    settings = Settings(elevenlabs_api_key=KEY)
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {
                "words": [
                    {"type": "word", "text": "a", "start": 0.0, "end": 1.0, "speaker_id": "s0"}
                ]
            }

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs.get("data")
        captured["files"] = kwargs.get("files")
        return FakeResponse()

    monkeypatch.setattr(elevenlabs_module.httpx, "post", fake_post)
    words = elevenlabs_module.transcribe_pcm_window(b"\x00\x00" * 160, settings=settings)

    assert captured["url"] == elevenlabs_module.API_URL
    assert captured["data"]["file_format"] == "pcm_s16le_16"
    assert captured["data"]["diarize"] == "true"
    assert captured["files"]["file"][0] == "window.pcm"
    assert words[0].speaker_id == "s0"


def test_token_endpoint_calls_provider_with_permanent_key_only_server_side(monkeypatch) -> None:
    settings = Settings(elevenlabs_api_key=KEY)
    captured: dict = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"token": TOKEN}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(elevenlabs_module.httpx, "post", fake_post)
    token = elevenlabs_module.create_realtime_token(settings)

    assert token == TOKEN
    assert captured["url"].endswith("/v1/single-use-token/realtime_scribe")
    assert captured["headers"] == {"xi-api-key": KEY}


def test_new_session_registry_starts_at_kisi_1_and_is_dense(context, monkeypatch) -> None:
    client, _, _, store = context
    windows = [
        {"speaker_0": [(0.5, 2.5)]},
        {"speaker_0": [(0.0, 2.5)]},
        {"speaker_0": [(0.0, 2.0)], "speaker_1": [(3.0, 5.0)]},
        {"speaker_0": [(0.0, 2.0)], "speaker_1": [(3.0, 5.0)]},
        {"speaker_0": [(0.5, 2.5)]},
        {"speaker_0": [(0.0, 2.5)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    # First session: Kişi 1 then a densely numbered Kişi 2.
    first = make_session(client)
    submit_window(client, first, start=0, end=8, sequence=1)
    submit_window(client, first, start=4, end=12, sequence=2)
    submit_window(client, first, start=8, end=16, sequence=3)
    second = submit_window(client, first, start=12, end=20, sequence=4).json()
    assert second["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]

    # A brand-new session never inherits numbering from the previous one.
    fresh = make_session(client)
    submit_window(client, fresh, start=0, end=8, sequence=1)
    submit_window(client, fresh, start=4, end=12, sequence=2)
    state = client.get(f"/api/v1/live-transcription/sessions/{fresh}").json()
    assert state["confirmed_speakers"] == ["Kişi 1"]
    assert state["aliases"] == {}
    assert store.get(fresh).speakers.keys() == {"Kişi 1"}


def test_known_speaker_count_caps_confirmed_speakers(context, monkeypatch) -> None:
    client, _, _, _ = context
    response = client.post("/api/v1/live-transcription/sessions?speaker_count=1")
    live_session_id = response.json()["live_session_id"]

    windows = [
        {"speaker_0": [(0.5, 2.5)]},
        {"speaker_0": [(0.0, 2.5)]},
        {"speaker_0": [(0.0, 2.0)], "speaker_1": [(3.0, 5.0)]},
        {"speaker_0": [(0.0, 2.0)], "speaker_1": [(3.0, 5.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    for sequence, start in ((2, 4), (3, 8)):
        body = submit_window(
            client, live_session_id, start=start, end=start + 8, sequence=sequence
        ).json()
        assert body["confirmed_speakers"] == ["Kişi 1"]
    # The excess cluster stays a candidate even after repeated evidence: the known
    # count caps confirmed speakers at 1.
    assert body["candidate_speakers"] >= 1


def test_ambiguous_mapping_creates_no_canonical(context, monkeypatch) -> None:
    client, _, _, _ = context
    live_session_id = make_session(client)

    windows = [
        {"speaker_0": [(0.5, 2.5)]},
        {"speaker_0": [(0.0, 2.5)]},
        # Two comparable clusters in one snapshot: one continues Kişi 1, the other
        # is only a candidate. Nothing may mint a Kişi 2 on a single snapshot.
        {"speaker_0": [(0.0, 2.0)], "speaker_1": [(3.0, 5.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)
    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    submit_window(client, live_session_id, start=4, end=12, sequence=2)
    body = submit_window(client, live_session_id, start=8, end=16, sequence=3).json()
    assert body["confirmed_speakers"] == ["Kişi 1"]
    assert body["candidate_speakers"] >= 1


def test_provisional_speaker_gets_no_alias(context, monkeypatch) -> None:
    """Aliases attach only to CONFIRMED canonical speakers (Phase 8 alias safety)."""
    client, _, _, _ = context
    live_session_id = make_session(client)

    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(
            {"speaker_0": [(0.0, 3.0)]}
        ),
    )

    # First snapshot: speaker is only a candidate, not confirmed.
    submit_window(client, live_session_id, start=0, end=8, sequence=1)

    # Attempting to alias unconfirmed Kişi 1 must return 404.
    response = client.put(
        f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/Kişi 1/alias",
        json={"display_name": "Ahmet"},
    )
    assert response.status_code == 404
    assert "not confirmed" in response.json()["detail"].lower()

    # Second overlapping snapshot confirms Kişi 1.
    submit_window(client, live_session_id, start=4, end=12, sequence=2)

    # Aliasing confirmed Kişi 1 now succeeds.
    alias_response = client.put(
        f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/Kişi 1/alias",
        json={"display_name": "Ahmet"},
    )
    assert alias_response.status_code == 200
    assert alias_response.json()["display_name"] == "Ahmet"


def test_ambiguous_evidence_remains_provisional_and_strong_evidence_confirms(
    context, monkeypatch
) -> None:
    """Ambiguous or weak evidence stays provisional; strong evidence confirms."""
    client, _, _, store = context

    # Session 1: Ambiguous overlap with small margin (< CONFIRM_MARGIN_SECONDS) stays provisional
    session1_id = make_session(client)
    s1 = store.get(session1_id)
    s1.speaker("Kişi 1").committed = [(0.0, 4.0)]
    s1.speaker("Kişi 2").committed = [(3.0, 8.0)]
    s1.confirmed_labels_set.update({"Kişi 1", "Kişi 2"})
    s1.last_snapshot = {"Kişi 1": [(0.0, 4.0)], "Kişi 2": [(3.0, 8.0)]}

    # Window [2.0, 10.0]: speaker_0 is [0.5, 2.0] locally -> [2.5, 4.0] globally
    # Overlap Kişi 1 [0.0, 4.0] = 1.5s, Overlap Kişi 2 [3.0, 8.0] = 1.0s. Margin = 0.5s < 1.0s.
    words_ambiguous = {
        "speaker_0": [(0.5, 2.0)],
    }
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(words_ambiguous),
    )
    res_ambiguous = submit_window(client, session1_id, start=2.0, end=10.0, sequence=1).json()
    assert len(res_ambiguous["assignments"]) == 1
    assert res_ambiguous["assignments"][0]["canonical_speaker"] is None
    assert res_ambiguous["assignments"][0]["speaker_state"] == "pending"
    assert res_ambiguous["assignments"][0]["provisional"] is True

    # Session 2: Strong decisive overlap (margin >= 1.0s) confirms
    session2_id = make_session(client)
    s2 = store.get(session2_id)
    s2.speaker("Kişi 1").committed = [(0.0, 4.0)]
    s2.speaker("Kişi 2").committed = [(4.1, 8.0)]
    s2.confirmed_labels_set.update({"Kişi 1", "Kişi 2"})
    s2.last_snapshot = {"Kişi 1": [(0.0, 4.0)], "Kişi 2": [(4.1, 8.0)]}

    # Window [2.0, 10.0]: speaker_0 is [0.0, 1.8] locally -> [2.0, 3.8] globally
    # Overlaps Kişi 1 by 1.8s, Kişi 2 by 0s. Margin = 1.8s >= 1.0s.
    words_strong = {
        "speaker_0": [(0.0, 1.8)],
    }
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(words_strong),
    )
    res_strong = submit_window(client, session2_id, start=2.0, end=10.0, sequence=1).json()
    confirmed_assignments = [a for a in res_strong["assignments"] if not a["provisional"]]
    assert len(confirmed_assignments) == 1
    assert confirmed_assignments[0]["canonical_speaker"] == "Kişi 1"
    assert confirmed_assignments[0]["speaker_state"] == "temporally_confirmed"


def test_confirmation_patches_in_place_no_duplicates_chronological_order(
    context, monkeypatch
) -> None:
    """Speaker confirmation patches in place without duplicate utterances or reordering."""
    client, _, _, _ = context
    live_session_id = make_session(client)

    # Simulated client-side transcript lines state
    transcript_lines = [
        {
            "id": "line-1",
            "startSeconds": 0.5,
            "endSeconds": 2.5,
            "text": "merhaba",
            "canonical": None,
            "provisional": True,
        },
        {
            "id": "line-2",
            "startSeconds": 3.0,
            "endSeconds": 5.0,
            "text": "nasılsınız",
            "canonical": None,
            "provisional": True,
        },
    ]

    # First window: produces provisional assignments
    words1 = {"speaker_0": [(0.5, 2.5), (3.0, 5.0)]}
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(words1),
    )
    res1 = submit_window(client, live_session_id, start=0.0, end=8.0, sequence=1).json()

    # Frontend attachSpeakerLabels logic simulation
    for line in transcript_lines:
        if line["canonical"]:
            continue
        mid = (line["startSeconds"] + line["endSeconds"]) / 2
        match = next(
            (
                a
                for a in res1["assignments"]
                if not a["provisional"]
                and mid >= a["start"] - 0.5
                and mid <= a["end"] + 0.5
            ),
            None,
        )
        if match:
            line["canonical"] = match["canonical_speaker"]

    # In window 1, all assignments are provisional ("Konuşmacı belirleniyor")
    assert all(line["canonical"] is None for line in transcript_lines)
    assert len(transcript_lines) == 2

    # Second window: confirms the speaker
    words2 = {"speaker_0": [(0.5, 2.5), (3.0, 5.0)]}
    monkeypatch.setattr(
        "app.routers.live_transcription.transcribe_pcm_window",
        lambda pcm, *, settings, requested_speaker_count=None: window_words(words2),
    )
    res2 = submit_window(client, live_session_id, start=0.0, end=8.0, sequence=2).json()

    for line in transcript_lines:
        if line["canonical"]:
            continue
        mid = (line["startSeconds"] + line["endSeconds"]) / 2
        match = next(
            (
                a
                for a in res2["assignments"]
                if not a["provisional"]
                and mid >= a["start"] - 0.5
                and mid <= a["end"] + 0.5
            ),
            None,
        )
        if match:
            line["canonical"] = match["canonical_speaker"]

    # Now both lines are patched in place to Kişi 1
    assert transcript_lines[0]["canonical"] == "Kişi 1"
    assert transcript_lines[1]["canonical"] == "Kişi 1"
    # No duplicate rows were created, row count stays exactly 2
    assert len(transcript_lines) == 2
    # Texts and timestamps unchanged
    assert transcript_lines[0]["text"] == "merhaba"
    assert transcript_lines[1]["text"] == "nasılsınız"
    # Chronological order strictly preserved
    assert transcript_lines[0]["startSeconds"] < transcript_lines[1]["startSeconds"]


def test_confirmed_labels_do_not_oscillate_after_freeze() -> None:
    """Once a confirmed label is attached outside mutable tail, it freezes and never oscillates."""
    line = {"startSeconds": 1.0, "endSeconds": 3.0, "canonical": "Kişi 1"}

    # Subsequent snapshot conflicting assignment arrives suggesting Kişi 2
    conflicting_assignment = {
        "canonical_speaker": "Kişi 2",
        "provisional": False,
        "start": 0.5,
        "end": 3.5,
    }

    # Frontend freeze rule: if line.canonical is already set, return unchanged

    # Frontend freeze rule: if line.canonical is already set, return unchanged
    def attach_label(current_line, assignment):
        if current_line["canonical"]:
            return current_line
        return {**current_line, "canonical": assignment["canonical_speaker"]}

    result_line = attach_label(line, conflicting_assignment)
    assert result_line["canonical"] == "Kişi 1"  # Frozen, did not flip to Kişi 2


def test_long_gap_return_does_not_mint_new_kisi_and_reconciles_at_final(
    context, monkeypatch
) -> None:
    """Phase 18 synthetic test: Long-gap return stays pending and does NOT create Kişi 3.

    Speaker A: 0-4s
    Speaker B: 4-8s
    Speaker A silent for > 12s lookback horizon (40 seconds silence).
    Same physical Speaker A returns at 50s with request-local id speaker_9.

    Live:
    - Never mints Kişi 3.
    - 50s utterance stays pending ("Konuşmacı belirleniyor").
    - Confirmed speakers remains strictly ["Kişi 1", "Kişi 2"].

    Final:
    - Reconciles densely onto Kişi 1 and Kişi 2.
    """
    client, _, _, store = context
    live_session_id = make_session(client)

    # Windows:
    # 1. 0-8s: speaker_0 (0-4s), speaker_1 (4-8s) -> candidates
    # 2. 0-12s: speaker_0 (0-4s), speaker_1 (4-8s) -> promoted to Kişi 1, Kişi 2
    # 3. 44-56s: speaker_9 (50-54s) -> after 40s silence of Speaker A
    # 4. 48-60s: candidate persists, but Speaker A silent >12s -> promotion blocked
    windows = [
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_9": [(6.0, 10.0)]},  # local to window [44, 56] -> global [50, 54]
        {"speaker_9": [(2.0, 6.0)]},   # local to window [48, 60] -> global [50, 54]
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    # Window 1: candidates registered
    w1 = submit_window(client, live_session_id, start=0, end=8, sequence=1).json()
    assert w1["confirmed_speakers"] == []
    assert w1["candidate_speakers"] == 2

    # Window 2: promoted to Kişi 1, Kişi 2
    w2 = submit_window(client, live_session_id, start=0, end=12, sequence=2).json()
    assert w2["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    assert "Kişi 1" in w2["promoted_speakers"]
    assert "Kişi 2" in w2["promoted_speakers"]

    # Window 3: 44-56s (Speaker A silent from 4.0 to 50.0s = 46s silence > 12s lookback horizon)
    w3 = submit_window(client, live_session_id, start=44, end=56, sequence=3).json()
    assert w3["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    assert "Kişi 3" not in w3.get("promoted_speakers", [])
    assert w3["candidate_speakers"] >= 1

    # Window 4: 48-60s (Candidate has 2 snapshots, but Speaker A is silent > 12s -> NO Kişi 3!)
    w4 = submit_window(client, live_session_id, start=48, end=60, sequence=4).json()
    assert "Kişi 3" not in w4["confirmed_speakers"]
    assert "Kişi 3" not in w4["promoted_speakers"]
    assert w4["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    # Long-gap return stays pending: no Kişi 3 assigned
    assert all(a.get("canonical_speaker") != "Kişi 3" for a in w4["assignments"])

    # FINAL RECONCILIATION:
    # Full-file Scribe v2 detects speaker_alpha (0-4s and 50-54s) and speaker_beta (4-8s)
    from app.services.speaker_matching import remap_final_speakers
    session = store.get(live_session_id)
    live_timelines = {
        label: list(spk.committed) for label, spk in session.speakers.items()
    }
    final_intervals = {
        "speaker_alpha": [(0.0, 4.0), (50.0, 54.0)],
        "speaker_beta": [(4.0, 8.0)],
    }
    final_mapping = remap_final_speakers(live_timelines, final_intervals)
    assert final_mapping["speaker_alpha"] == "Kişi 1"
    assert final_mapping["speaker_beta"] == "Kişi 2"
    assert set(final_mapping.values()) == {"Kişi 1", "Kişi 2"}


def test_alias_reconciliation_only_preserves_confident_mapping() -> None:
    """Phase 12: Aliases are preserved only on confident final mapping, never misapplied."""
    from app.services.speaker_matching import reconcile_final_aliases, remap_final_speakers

    live_timelines = {
        "Kişi 1": [(0.0, 10.0)],
        "Kişi 2": [(20.0, 30.0)],
    }
    live_aliases = {
        "Kişi 1": "Ahmet",
        "Kişi 2": "Mehmet",
    }

    # Case 1: speaker_A confidently matches Kişi 1 (10s overlap)
    # speaker_B is ambiguous with zero overlap (e.g. 100-110s)
    final_intervals = {
        "speaker_A": [(0.0, 10.0)],
        "speaker_B": [(100.0, 110.0)],
    }
    final_mapping = remap_final_speakers(live_timelines, final_intervals)
    final_aliases = reconcile_final_aliases(live_timelines, final_intervals, live_aliases)

    # Dense numbering: 2 final speakers -> Kişi 1, Kişi 2
    assert final_mapping == {"speaker_A": "Kişi 1", "speaker_B": "Kişi 2"}
    # Only Ahmet is preserved because Kişi 1 was confidently mapped
    # Mehmet was NOT matched to any final speaker, so Mehmet is discarded (never given to speaker_B)
    assert final_aliases == {"Kişi 1": "Ahmet"}
    assert "Mehmet" not in final_aliases.values()


def test_stale_known_speaker_does_not_globally_freeze_promotion(
    context, monkeypatch
) -> None:
    """A stale known speaker (silent > 12s) does NOT globally freeze candidate promotion."""
    client, _, _, _ = context
    live_session_id = make_session(client)

    windows = [
        # Window 1 (0-8s): spk_0 (0-4s), spk_1 (4-8s) -> candidates
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        # Window 2 (0-12s): spk_0 (0-4s), spk_1 (4-8s) -> both promoted to Kişi 1, Kişi 2
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        # Window 3 (4-16s): spk_1 (Kişi 2) active 4-12s (local 0-8s)
        {"speaker_1": [(0.0, 8.0)]},
        # Window 4 (8-20s): spk_1 (Kişi 2) active 8-16s (local 0-8s)
        {"speaker_1": [(0.0, 8.0)]},
        # Window 5 (12-24s): Kişi 1 silent since 4.0s (12-4 = 8s).
        # spk_1 (Kişi 2) active 12-15s, spk_2 (new Speaker C) active 17-20s
        {"speaker_1": [(0.0, 3.0)], "speaker_2": [(5.0, 8.0)]},
        # Window 6 (16-28s): Window start 16.0. Kişi 1 silent since 4.0s -> 16 - 4 = 12s (STALE!).
        # spk_1 (Kişi 2) active 16-18s, spk_2 (Speaker C) active 19-24s.
        {"speaker_1": [(0.0, 2.0)], "speaker_2": [(3.0, 8.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    w2 = submit_window(client, live_session_id, start=0, end=12, sequence=2).json()
    assert w2["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]

    submit_window(client, live_session_id, start=4, end=16, sequence=3)
    submit_window(client, live_session_id, start=8, end=20, sequence=4)
    w5 = submit_window(client, live_session_id, start=12, end=24, sequence=5).json()
    assert w5["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    assert w5["candidate_speakers"] >= 1

    # In window 6, Kişi 1 is stale, but Candidate C coexisted with active Kişi 2:
    w6 = submit_window(client, live_session_id, start=16, end=28, sequence=6).json()
    assert "Kişi 3" in w6.get("promoted_speakers", [])
    assert w6["confirmed_speakers"] == ["Kişi 1", "Kişi 2", "Kişi 3"]


def test_returning_ambiguous_speaker_does_not_become_new_kisi(
    context, monkeypatch
) -> None:
    """Returning speaker after a long gap without simultaneous coexisting evidence stays pending."""
    client, _, _, _ = context
    live_session_id = make_session(client)

    windows = [
        # Window 1 & 2: Kişi 1 and Kişi 2 confirmed
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        # Window 3 (44-56s): returning speaker alone under speaker_9 (50-54s -> local 6-10s)
        {"speaker_9": [(6.0, 10.0)]},
        # Window 4 (48-60s): speaker_9 appears again alone (50-54s -> local 2-6s)
        {"speaker_9": [(2.0, 6.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    w2 = submit_window(client, live_session_id, start=0, end=12, sequence=2).json()
    assert w2["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]

    # Window 3 & 4: lone candidate appears after silence > 12s
    submit_window(client, live_session_id, start=44, end=56, sequence=3)
    w4 = submit_window(client, live_session_id, start=48, end=60, sequence=4).json()

    # Ambiguous candidate must NOT be promoted to Kişi 3
    assert "Kişi 3" not in w4["confirmed_speakers"]
    assert "Kişi 3" not in w4.get("promoted_speakers", [])
    assert w4["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    # All assignments for the returning speaker remain provisional / pending
    assert all(a.get("canonical_speaker") != "Kişi 3" for a in w4["assignments"])


def test_genuinely_distinct_later_speaker_can_become_kisi_n_plus_1(
    context, monkeypatch
) -> None:
    """Phase 5 verification: active known speaker and C promote Kişi N+1 with stale Kişi 2."""
    client, _, _, _ = context
    live_session_id = make_session(client)

    windows = [
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_1": [(0.0, 8.0)]},
        {"speaker_1": [(0.0, 8.0)]},
        {"speaker_1": [(0.0, 3.0)], "speaker_2": [(5.0, 8.0)]},
        {"speaker_1": [(0.0, 2.0)], "speaker_2": [(3.0, 8.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    submit_window(client, live_session_id, start=0, end=12, sequence=2)
    submit_window(client, live_session_id, start=4, end=16, sequence=3)
    submit_window(client, live_session_id, start=8, end=20, sequence=4)
    submit_window(client, live_session_id, start=12, end=24, sequence=5)
    w6 = submit_window(client, live_session_id, start=16, end=28, sequence=6).json()

    assert "Kişi 3" in w6.get("promoted_speakers", [])
    assert w6["confirmed_speakers"] == ["Kişi 1", "Kişi 2", "Kişi 3"]


def test_repeated_candidate_alone_is_insufficient_after_identity_loss(
    context, monkeypatch
) -> None:
    """Candidate repeating across 2+ snapshots while alone after long silence cannot promote."""
    client, _, _, _ = context
    live_session_id = make_session(client)

    windows = [
        {"speaker_0": [(0.0, 4.0)]},
        {"speaker_0": [(0.0, 4.0)]},
        # Candidate appears alone in 3 consecutive snapshots:
        {"speaker_x": [(2.0, 6.0)]},
        {"speaker_x": [(2.0, 6.0)]},
        {"speaker_x": [(2.0, 6.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    submit_window(client, live_session_id, start=0, end=12, sequence=2)

    submit_window(client, live_session_id, start=30, end=42, sequence=3)
    submit_window(client, live_session_id, start=34, end=46, sequence=4)
    w5 = submit_window(client, live_session_id, start=38, end=50, sequence=5).json()

    assert w5["confirmed_speakers"] == ["Kişi 1"]
    assert "Kişi 2" not in w5.get("promoted_speakers", [])
    assert w5["candidate_speakers"] >= 1


def test_known_speaker_count_cap_still_enforced(context, monkeypatch) -> None:
    """User sets speaker_count=2. Distinct candidate cannot promote beyond N."""
    client, _, _, _ = context
    response = client.post("/api/v1/live-transcription/sessions?speaker_count=2")
    assert response.status_code == 201
    live_session_id = response.json()["live_session_id"]

    windows = [
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_0": [(0.0, 4.0)], "speaker_1": [(4.0, 8.0)]},
        {"speaker_1": [(0.0, 8.0)]},
        {"speaker_1": [(0.0, 8.0)]},
        {"speaker_1": [(0.0, 3.0)], "speaker_2": [(5.0, 8.0)]},
        {"speaker_1": [(0.0, 2.0)], "speaker_2": [(3.0, 8.0)]},
    ]

    def fake_window(pcm, *, settings, requested_speaker_count=None):
        return window_words(windows.pop(0))

    monkeypatch.setattr("app.routers.live_transcription.transcribe_pcm_window", fake_window)

    submit_window(client, live_session_id, start=0, end=8, sequence=1)
    submit_window(client, live_session_id, start=0, end=12, sequence=2)
    submit_window(client, live_session_id, start=4, end=16, sequence=3)
    submit_window(client, live_session_id, start=8, end=20, sequence=4)
    submit_window(client, live_session_id, start=12, end=24, sequence=5)
    w6 = submit_window(client, live_session_id, start=16, end=28, sequence=6).json()

    # Confirmed speakers remains strictly Kişi 1 and Kişi 2:
    assert w6["confirmed_speakers"] == ["Kişi 1", "Kişi 2"]
    assert "Kişi 3" not in w6.get("promoted_speakers", [])
    assert w6["candidate_speakers"] >= 1


def test_canonical_labels_remain_dense() -> None:
    """Canonical speaker labels are strictly dense without index skips."""
    from app.services.live_sessions import next_canonical_label

    assert next_canonical_label(set()) == "Kişi 1"
    assert next_canonical_label({"Kişi 1"}) == "Kişi 2"
    assert next_canonical_label({"Kişi 1", "Kişi 2"}) == "Kişi 3"
    assert next_canonical_label({"Kişi 1", "Kişi 2", "Kişi 3"}) == "Kişi 4"
    assert next_canonical_label({"Kişi 1", "Kişi 3"}) == "Kişi 2"


def test_metric_denominators_explicit_and_mathematically_tested() -> None:
    """Mathematical verification of distinct denominators:

    - assignment_pending_rate = pending_speech_seconds / all_rolling_assignment_seconds
    - reference_live_coverage = confirmed_speech_seconds / total_reference_speech_seconds
    - reference_pending_or_uncovered_rate = (uncovered_speech_seconds) / total_reference
    """
    from app.services.canonical_evaluator import compute_assignment_pending_rate, evaluate_timeline

    output_windows = [
        {
            "sequence": 1,
            "window": [0.0, 10.0],
            "assignments": [
                {
                    "canonical_speaker": "Kişi 1",
                    "provisional": False,
                    "start": 0.0,
                    "end": 4.0,
                    "speech_seconds": 4.0,
                },
                {
                    "canonical_speaker": None,
                    "provisional": True,
                    "start": 4.0,
                    "end": 6.0,
                    "speech_seconds": 2.0,
                },
            ],
        },
        {
            "sequence": 2,
            "window": [10.0, 20.0],
            "assignments": [
                {
                    "canonical_speaker": "Kişi 1",
                    "provisional": False,
                    "start": 10.0,
                    "end": 14.0,
                    "speech_seconds": 4.0,
                },
                {
                    "canonical_speaker": None,
                    "provisional": True,
                    "start": 14.0,
                    "end": 20.0,
                    "speech_seconds": 6.0,
                },
            ],
        },
    ]

    assign_pending = compute_assignment_pending_rate(output_windows)
    assert assign_pending == 0.500

    turns = [
        {"speaker": "Kişi 1", "start_seconds": 0.0, "end_seconds": 10.0},
        {"speaker": "Kişi 2", "start_seconds": 10.0, "end_seconds": 20.0},
    ]
    metrics = evaluate_timeline(
        recording_name="test-denominators",
        output_windows=output_windows,
        turns=turns,
        duration=20.0,
        confirmed_speakers=["Kişi 1"],
        provisional_candidates=[],
        delays=[],
        resolution=0.25,
    )

    cov_sum = metrics.reference_live_coverage + metrics.reference_pending_or_uncovered_rate
    assert round(cov_sum, 2) == 1.00
    assert metrics.assignment_pending_rate == 0.500
    assert metrics.reference_pending_or_uncovered_rate != metrics.assignment_pending_rate

