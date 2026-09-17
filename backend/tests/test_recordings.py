import subprocess
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.ffmpeg import FFmpegConversionError
from app.services.recordings import MP3_FILENAME, WAV_FILENAME
from tests.conftest import upload


def converted_files(meetings_dir: Path, recording_id: str) -> list[str]:
    return sorted(path.name for path in (meetings_dir / recording_id).iterdir())


def probe_with_ffprobe(path: Path, entries: str) -> str:
    completed = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            entries,
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@pytest.mark.parametrize(
    ("sample_key", "mime_type", "filename"),
    [
        ("webm", "audio/webm;codecs=opus", "recording.webm"),
        ("mp4", "audio/mp4", "recording.mp4"),
    ],
)
def test_valid_audio_upload_succeeds(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
    sample_key: str,
    mime_type: str,
    filename: str,
) -> None:
    payload = audio_samples[sample_key].read_bytes()

    response = upload(
        client,
        payload,
        mime_type=mime_type,
        filename=filename,
    )

    assert response.status_code == 201, response.text
    body = response.json()

    assert body["input"]["size_bytes"] == len(payload)
    assert body["input"]["mime_type"] == mime_type
    assert body["mp3"]["size_bytes"] > 0
    assert body["processing_wav"]["sample_rate"] == 16000
    assert body["processing_wav"]["channels"] == 1
    assert body["conversion_ms"] >= 0
    assert 1.5 <= body["duration_seconds"] <= 2.5

    recording_dir = meetings_dir / body["recording_id"]
    assert {MP3_FILENAME, WAV_FILENAME} <= set(converted_files(meetings_dir, body["recording_id"]))
    assert (recording_dir / MP3_FILENAME).stat().st_size == body["mp3"]["size_bytes"]
    assert (recording_dir / WAV_FILENAME).stat().st_size == body["processing_wav"]["size_bytes"]


def test_processing_wav_is_16khz_mono_pcm_s16le(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
) -> None:
    response = upload(client, audio_samples["webm"].read_bytes())
    recording_id = response.json()["recording_id"]
    wav_path = meetings_dir / recording_id / WAV_FILENAME

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getcomptype() == "NONE"
        duration = wav_file.getnframes() / wav_file.getframerate()

    assert 1.5 <= duration <= 2.5
    probed = probe_with_ffprobe(wav_path, "stream=codec_name,sample_rate,channels")
    assert probed == "pcm_s16le,16000,1"


def test_meeting_mp3_is_valid_mono_mp3(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
) -> None:
    response = upload(client, audio_samples["webm"].read_bytes())
    recording_id = response.json()["recording_id"]

    codec = probe_with_ffprobe(meetings_dir / recording_id / MP3_FILENAME, "stream=codec_name")
    assert codec == "mp3"


def test_temporary_source_file_is_removed_after_success(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
) -> None:
    response = upload(client, audio_samples["webm"].read_bytes())
    recording_id = response.json()["recording_id"]

    files = converted_files(meetings_dir, recording_id)
    assert files == [MP3_FILENAME, WAV_FILENAME]


def test_unsupported_mime_type_is_rejected(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
) -> None:
    response = upload(
        client,
        audio_samples["webm"].read_bytes(),
        mime_type="video/webm",
        filename="recording.webm",
        content_type="video/webm",
    )

    assert response.status_code == 415
    assert list(meetings_dir.iterdir()) == []


def test_non_audio_content_is_rejected(
    client: TestClient,
    meetings_dir: Path,
) -> None:
    response = upload(client, b"this is definitely not audio", filename="fake.webm")

    assert response.status_code == 422
    assert list(meetings_dir.iterdir()) == []


def test_empty_upload_is_rejected(
    client: TestClient,
    meetings_dir: Path,
) -> None:
    response = upload(client, b"")

    assert response.status_code == 422
    assert list(meetings_dir.iterdir()) == []


def test_filename_path_traversal_is_ignored(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
) -> None:
    response = upload(
        client,
        audio_samples["webm"].read_bytes(),
        filename="../../evil.webm",
    )

    assert response.status_code == 201, response.text
    recording_id = response.json()["recording_id"]

    assert recording_id.isalnum()
    assert not (meetings_dir.parent / "evil.webm").exists()
    assert converted_files(meetings_dir, recording_id) == [MP3_FILENAME, WAV_FILENAME]


def test_conversion_failure_leaves_no_partial_outputs(
    client: TestClient,
    meetings_dir: Path,
    audio_samples: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_conversion(*args: object, **kwargs: object) -> int:
        raise FFmpegConversionError("simulated ffmpeg failure")

    monkeypatch.setattr("app.services.recordings.convert_to_mp3_and_wav", failing_conversion)

    response = upload(client, audio_samples["webm"].read_bytes())

    assert response.status_code == 422
    assert list(meetings_dir.iterdir()) == []
