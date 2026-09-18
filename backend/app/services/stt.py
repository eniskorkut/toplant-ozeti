"""Production STT service: whisper.cpp v1.9.4 with JSON-full timestamp output.

Only the validated path is ported: large-v3-turbo Q8_0, CPU, `-ojf` timestamps and
no `-nt` (disabling timestamps changed the decoded text in the benchmark round).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


class SttError(RuntimeError):
    """Raised when transcription fails."""


class SttModelMissingError(SttError):
    """Raised when the whisper.cpp binary or model file is unavailable."""


@dataclass(frozen=True)
class SttWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SttSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SttResult:
    words: list[SttWord]
    segments: list[SttSegment]
    text: str
    language: str | None
    inference_seconds: float


def _resolve_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise SttModelMissingError(f"whisper.cpp binary {binary!r} was not found on PATH")
    return resolved


def parse_whisper_json(payload: dict) -> tuple[list[SttWord], list[SttSegment], str]:
    """Aggregate word-piece tokens into words at space boundaries."""
    words: list[SttWord] = []
    segments: list[SttSegment] = []
    current: dict | None = None

    for segment in payload.get("transcription", []):
        segments.append(
            SttSegment(
                start=segment["offsets"]["from"] / 1000.0,
                end=segment["offsets"]["to"] / 1000.0,
                text=segment["text"].strip(),
            )
        )
        for token in segment.get("tokens", []):
            text = token["text"]
            if text.startswith("[_"):
                continue
            start = token["offsets"]["from"] / 1000.0
            end = token["offsets"]["to"] / 1000.0
            if text.startswith(" ") or current is None:
                if current is not None:
                    words.append(SttWord(current["start"], current["end"], current["text"]))
                current = {"start": start, "end": end, "text": text.strip()}
            else:
                current["end"] = end
                current["text"] = f"{current['text']}{text.strip()}"

    if current is not None:
        words.append(SttWord(current["start"], current["end"], current["text"]))

    text = " ".join(segment.text for segment in segments).strip()
    return words, segments, text


def transcribe(audio_path: Path, settings: Settings) -> SttResult:
    binary = _resolve_binary(settings.whisper_cpp_binary)
    model_path = settings.whisper_cpp_model
    if not model_path.exists():
        raise SttModelMissingError(f"whisper.cpp model not found: {model_path}")

    with tempfile.TemporaryDirectory(prefix="meeting-stt-") as temp_dir:
        output_prefix = Path(temp_dir) / "transcript"
        command = [
            binary,
            "-m",
            str(model_path),
            "-f",
            str(audio_path),
            "-l",
            settings.stt_language,
            "-t",
            str(settings.stt_threads),
            "-bs",
            str(settings.stt_beam_size),
            "-ojf",
            "-of",
            str(output_prefix),
            "-np",
        ]

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=settings.stt_timeout_seconds,
                check=False,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as exc:
            raise SttError(
                f"whisper.cpp timed out after {settings.stt_timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise SttError(f"failed to execute whisper.cpp: {exc}") from exc
        inference_seconds = time.perf_counter() - started

        if completed.returncode != 0:
            tail = "\n".join((completed.stderr or "").strip().splitlines()[-10:])
            raise SttError(
                f"whisper.cpp exited with code {completed.returncode}: {tail or 'no output'}"
            )

        json_path = output_prefix.with_suffix(".json")
        if not json_path.exists():
            raise SttError("whisper.cpp did not produce a JSON transcript")
        payload = json.loads(json_path.read_text(encoding="utf-8"))

    words, segments, text = parse_whisper_json(payload)
    language = (payload.get("result") or {}).get("language")
    return SttResult(
        words=words,
        segments=segments,
        text=text,
        language=language,
        inference_seconds=round(inference_seconds, 3),
    )
