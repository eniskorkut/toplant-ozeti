from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Runtime configuration for the Meeting Intelligence backend."""

    model_config = SettingsConfigDict(env_prefix="MEETING_", env_file=".env", extra="ignore")

    app_name: str = "Meeting Intelligence API"
    app_version: str = "0.1.0"

    # Application log level (standard library logging, no extra dependency).
    log_level: str = "INFO"

    # Storage: meeting audio and generated artifacts stay on the local filesystem.
    # Never committed to git (see .gitignore).
    data_dir: Path = REPO_ROOT / "data" / "meetings"

    # Resolved through the system PATH on purpose: no platform-specific paths.
    ffmpeg_binary: str = "ffmpeg"

    # Centralized CORS configuration for the host-native frontend during
    # development. Never use "*" here.
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3100",
        "http://127.0.0.1:3100",
    ]

    # Upper bound for a single ffmpeg conversion run.
    ffmpeg_timeout_seconds: int = 1800

    # --- processing pipeline -------------------------------------------------
    # SQLite database shared by the API process and the worker process.
    database_url: str = "sqlite+aiosqlite:////data/app.db"

    # whisper.cpp (production STT runtime; v1.9.4, large-v3-turbo Q8_0).
    whisper_cpp_binary: str = "whisper-cli"
    whisper_cpp_model: Path = Path("/models/whisper/ggml-large-v3-turbo-q8_0.bin")
    stt_threads: int = 8
    stt_language: str = "auto"
    stt_beam_size: int = 5
    stt_timeout_seconds: int = 1800

    # sherpa-onnx diarization (pyannote 3.0 + TitaNet Small).
    diarization_segmentation_model: Path = Path(
        "/models/diarization/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    )
    diarization_embedding_model: Path = Path(
        "/models/diarization/nemo_en_titanet_small.onnx"
    )
    diarization_threads: int = 8
    diarization_threshold: float = 0.80
    diarization_min_duration_on: float = 0.3
    diarization_min_duration_off: float = 0.5

    # Merge: boundary tolerance validated in the benchmark round.
    merge_boundary_tolerance_seconds: float = 0.25

    # Worker loop.
    worker_poll_seconds: float = 2.0

    @property
    def meetings_dir(self) -> Path:
        return self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
