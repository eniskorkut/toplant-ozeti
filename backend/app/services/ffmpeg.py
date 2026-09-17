import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FFMPEG_BINARY = "ffmpeg"
DEFAULT_FFPROBE_BINARY = "ffprobe"
_TIMEOUT_SECONDS = 10
_PROBE_TIMEOUT_SECONDS = 60


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg executable cannot be located on the system PATH."""


class FFmpegError(RuntimeError):
    """Base class for ffmpeg/ffprobe execution failures."""


class FFprobeError(FFmpegError):
    """Raised when ffprobe cannot inspect the given file."""


class FFmpegConversionError(FFmpegError):
    """Raised when the ffmpeg conversion process fails."""


@dataclass(frozen=True)
class FFmpegInfo:
    path: str
    version: str


@dataclass(frozen=True)
class AudioStreamInfo:
    """What ffprobe reports for the first audio stream of a file."""

    codec_name: str
    sample_rate: int | None
    channels: int | None
    duration_seconds: float | None


def find_ffmpeg(binary: str = DEFAULT_FFMPEG_BINARY) -> str | None:
    """Locate ffmpeg using the system PATH (no hard-coded platform paths)."""
    return shutil.which(binary)


def is_ffmpeg_available(binary: str = DEFAULT_FFMPEG_BINARY) -> bool:
    return find_ffmpeg(binary) is not None


def parse_ffmpeg_version(output: str) -> str:
    """Extract the version token from the first line of `ffmpeg -version`."""
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    parts = first_line.split()
    if len(parts) >= 3 and parts[0] == "ffmpeg" and parts[1] == "version":
        return parts[2]
    return first_line or "unknown"


def get_ffmpeg_info(binary: str = DEFAULT_FFMPEG_BINARY) -> FFmpegInfo:
    """Return the ffmpeg path and version, or raise FFmpegNotFoundError."""
    path = find_ffmpeg(binary)
    if path is None:
        raise FFmpegNotFoundError(f"{binary!r} was not found on the system PATH")

    try:
        completed = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        raise FFmpegNotFoundError(f"Failed to execute {path!r}: {exc}") from exc

    output = completed.stdout or completed.stderr
    return FFmpegInfo(path=path, version=parse_ffmpeg_version(output))


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def probe_audio_stream(
    path: Path,
    binary: str = DEFAULT_FFPROBE_BINARY,
    timeout_seconds: int = _PROBE_TIMEOUT_SECONDS,
) -> AudioStreamInfo | None:
    """Inspect the first audio stream of a file.

    Returns None when the file contains no audio stream, and raises FFprobeError
    when the file cannot be parsed at all. Never trusts the file extension or the
    client-provided MIME type.
    """
    command = [
        binary,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFprobeError(f"Failed to execute ffprobe: {exc}") from exc

    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-10:])
        raise FFprobeError(f"ffprobe rejected the file: {tail or 'unknown error'}")

    try:
        data = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFprobeError(f"ffprobe produced unreadable output: {exc}") from exc

    streams = data.get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    duration = _as_float(stream.get("duration"))
    if duration is None:
        duration = _as_float((data.get("format") or {}).get("duration"))

    return AudioStreamInfo(
        codec_name=str(stream.get("codec_name") or "unknown"),
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        duration_seconds=duration,
    )


def convert_to_mp3_and_wav(
    source: Path,
    mp3_path: Path,
    wav_path: Path,
    binary: str = DEFAULT_FFMPEG_BINARY,
    timeout_seconds: int = 1800,
) -> int:
    """Decode `source` once and write both outputs in a single ffmpeg process.

    - mp3_path: mono MP3 (~96 kbps) for playback/archive
    - wav_path: mono 16 kHz signed 16-bit PCM for STT and diarization

    Returns the conversion wall-clock time in milliseconds. Raises
    FFmpegConversionError when ffmpeg exits unsuccessfully.
    """
    command = [
        binary,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "96k",
        "-ac",
        "1",
        str(mp3_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffmpeg timed out after {timeout_seconds}s") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise FFmpegError(f"Failed to execute ffmpeg: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - start) * 1000)

    if completed.returncode != 0:
        tail = "\n".join((completed.stderr or "").strip().splitlines()[-20:])
        raise FFmpegConversionError(
            f"ffmpeg exited with code {completed.returncode}: {tail or 'unknown error'}"
        )

    return elapsed_ms
