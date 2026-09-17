import shutil
import subprocess
from dataclasses import dataclass

DEFAULT_FFMPEG_BINARY = "ffmpeg"
_TIMEOUT_SECONDS = 10


class FFmpegNotFoundError(RuntimeError):
    """Raised when the ffmpeg executable cannot be located on the system PATH."""


@dataclass(frozen=True)
class FFmpegInfo:
    path: str
    version: str


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
