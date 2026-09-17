"""Audio condition measurement for the diarization benchmark (read-only).

Uses the application container's ffmpeg via `docker compose exec`, exactly like
the STT harness. The audio file is never modified.
"""

from __future__ import annotations

import re
import subprocess
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "meetings"

MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[\d.]+) dB")
MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?[\d.]+) dB")


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def measure(path: Path) -> dict:
    relative = path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    completed = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "backend", "ffmpeg",
            "-hide_banner", "-nostdin", "-i", f"/data/meetings/{relative}",
            "-af", "volumedetect", "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    max_match = MAX_VOLUME_RE.search(completed.stderr)
    mean_match = MEAN_VOLUME_RE.search(completed.stderr)
    maximum = float(max_match.group(1)) if max_match else None

    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "duration_seconds": round(duration_seconds(path), 3),
        "size_bytes": path.stat().st_size,
        "mean_volume_db": float(mean_match.group(1)) if mean_match else None,
        "max_volume_db": maximum,
        "clipping": bool(maximum is not None and maximum >= -0.1),
    }
