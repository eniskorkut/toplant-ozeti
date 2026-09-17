import logging
import shutil
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from app.config import Settings
from app.services.ffmpeg import (
    FFprobeError,
    convert_to_mp3_and_wav,
    probe_audio_stream,
)

logger = logging.getLogger(__name__)

# Containers produced by the browser MediaRecorder that we accept. The file
# extension is derived from this map, never from the client-provided filename.
ALLOWED_AUDIO_TYPES: dict[str, str] = {
    "audio/webm": "webm",
    "audio/webm;codecs=opus": "webm",
    "audio/mp4": "mp4",
}

SOURCE_STEM = "source"
MP3_FILENAME = "meeting.mp3"
WAV_FILENAME = "processing.wav"
COPY_BUFFER_BYTES = 1024 * 1024


class UnsupportedAudioTypeError(ValueError):
    """Raised when the declared MIME type is not one the pipeline accepts."""


class InvalidAudioFileError(ValueError):
    """Raised when the uploaded file is empty or is not decodable audio."""


@dataclass(frozen=True)
class RecordingArtifacts:
    recording_id: str
    duration_seconds: float
    input_mime_type: str
    input_size_bytes: int
    mp3_size_bytes: int
    wav_size_bytes: int
    wav_sample_rate: int
    wav_channels: int
    wav_sample_width_bytes: int
    conversion_ms: int


def normalize_mime_type(raw_mime_type: str) -> str:
    """Normalize a MIME type for comparison: lowercase, no whitespace padding."""
    return ";".join(part.strip().lower() for part in raw_mime_type.split(";") if part.strip())


def extension_for_mime_type(raw_mime_type: str) -> tuple[str, str]:
    """Return (normalized MIME type, safe file extension) or raise."""
    normalized = normalize_mime_type(raw_mime_type)
    extension = ALLOWED_AUDIO_TYPES.get(normalized)
    if extension is None:
        raise UnsupportedAudioTypeError(f"Unsupported audio MIME type: {raw_mime_type!r}")
    return normalized, extension


def copy_upload_to_disk(upload_file: BinaryIO, destination: Path) -> int:
    """Stream an upload to disk in fixed-size chunks; never buffers it fully."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        shutil.copyfileobj(upload_file, target, length=COPY_BUFFER_BYTES)
    return destination.stat().st_size


def read_wav_metadata(path: Path) -> tuple[int, int, int, float]:
    """Read sample rate, channels, sample width and duration from a WAV file."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
    duration_seconds = frames / sample_rate if sample_rate else 0.0
    return sample_rate, channels, sample_width, duration_seconds


def process_recording(
    *,
    upload_file: BinaryIO,
    mime_type: str,
    settings: Settings,
) -> RecordingArtifacts:
    """Store, validate and convert one browser recording.

    Layout: <meetings_dir>/<recording_id>/{source.<ext>, meeting.mp3, processing.wav}
    The temporary source file is removed on success; on failure the whole
    recording directory is removed so no orphan or partial files remain.
    """
    normalized_mime_type, extension = extension_for_mime_type(mime_type)

    recording_id = uuid.uuid4().hex
    recording_dir = settings.meetings_dir / recording_id
    source_path = recording_dir / f"{SOURCE_STEM}.{extension}"
    mp3_path = recording_dir / MP3_FILENAME
    wav_path = recording_dir / WAV_FILENAME

    try:
        input_size_bytes = copy_upload_to_disk(upload_file, source_path)
        if input_size_bytes == 0:
            raise InvalidAudioFileError("Uploaded file is empty")

        try:
            stream_info = probe_audio_stream(source_path)
        except FFprobeError as exc:
            raise InvalidAudioFileError("Uploaded file is not decodable audio") from exc
        if stream_info is None:
            raise InvalidAudioFileError("Uploaded file contains no audio stream")

        conversion_ms = convert_to_mp3_and_wav(
            source_path,
            mp3_path,
            wav_path,
            timeout_seconds=settings.ffmpeg_timeout_seconds,
        )
        sample_rate, channels, sample_width, duration_seconds = read_wav_metadata(wav_path)
    except Exception:
        shutil.rmtree(recording_dir, ignore_errors=True)
        raise

    source_path.unlink(missing_ok=True)

    logger.info(
        "stored recording %s (mime=%s, input=%d bytes, source_codec=%s, seconds=%.3f)",
        recording_id,
        normalized_mime_type,
        input_size_bytes,
        stream_info.codec_name,
        duration_seconds,
    )

    return RecordingArtifacts(
        recording_id=recording_id,
        duration_seconds=duration_seconds,
        input_mime_type=normalized_mime_type,
        input_size_bytes=input_size_bytes,
        mp3_size_bytes=mp3_path.stat().st_size,
        wav_size_bytes=wav_path.stat().st_size,
        wav_sample_rate=sample_rate,
        wav_channels=channels,
        wav_sample_width_bytes=sample_width,
        conversion_ms=conversion_ms,
    )
