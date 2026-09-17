import pytest

from app.services.ffmpeg import (
    FFmpegNotFoundError,
    find_ffmpeg,
    get_ffmpeg_info,
    is_ffmpeg_available,
    parse_ffmpeg_version,
)

SAMPLE_OUTPUT = (
    "ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers\n"
    "built with Apple clang version 16.0.0\n"
)


def test_parse_ffmpeg_version() -> None:
    assert parse_ffmpeg_version(SAMPLE_OUTPUT) == "7.1.1"


def test_parse_ffmpeg_version_handles_git_builds() -> None:
    assert parse_ffmpeg_version("ffmpeg version n7.1 Copyright (c)\n") == "n7.1"


def test_parse_ffmpeg_version_without_output() -> None:
    assert parse_ffmpeg_version("") == "unknown"


def test_find_ffmpeg_uses_system_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ffmpeg.shutil.which", lambda _binary: "/usr/local/bin/ffmpeg")

    assert find_ffmpeg() == "/usr/local/bin/ffmpeg"


def test_get_ffmpeg_info_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ffmpeg.shutil.which", lambda _binary: None)

    with pytest.raises(FFmpegNotFoundError):
        get_ffmpeg_info()


def test_is_ffmpeg_available_does_not_raise() -> None:
    assert isinstance(is_ffmpeg_available(), bool)
