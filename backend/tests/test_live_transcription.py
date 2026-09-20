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
        {"speaker_1": [(0.2, 0.8)]},
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

    assert output["turns"][0]["speaker"] == "Kişi 2"  # unseen final speaker: new label
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
