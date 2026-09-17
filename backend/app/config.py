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

    # Storage: meeting audio and generated artifacts stay on the local filesystem.
    # Never committed to git (see .gitignore).
    data_dir: Path = REPO_ROOT / "data" / "meetings"

    # Resolved through the system PATH on purpose: no platform-specific paths.
    ffmpeg_binary: str = "ffmpeg"

    @property
    def meetings_dir(self) -> Path:
        return self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
