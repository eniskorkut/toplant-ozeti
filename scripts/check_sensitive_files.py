#!/usr/bin/env python3
"""Pre-commit safety gate: refuse to commit secrets, audio, runtime data or raw outputs.

Standard library only. Inspects the files Git actually tracks plus whatever is currently
staged, and fails non-zero when a suspicious artifact or credential value is found.

Output is limited to `path: reason` lines — a detected secret value is never printed.

Usage:
    python scripts/check_sensitive_files.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- path rules -------------------------------------------------------------

ENV_ALLOWLIST = {".env.example", ".env.local.example", ".env.sample"}

AUDIO_EXTENSIONS = {".wav", ".mp3", ".webm", ".m4a"}
MODEL_EXTENSIONS = {".onnx", ".bin", ".gguf", ".pt", ".pth", ".safetensors"}
DATABASE_SUFFIXES = (".db", ".db-wal", ".db-shm")

# Ordered: a path under both private/ and results/ reports the more specific reason.
SENSITIVE_DIRECTORIES = ("private", "raw", "results")

# Safe aggregate/model metadata that is intentionally tracked inside results dirs.
SAFE_RESULTS_FILES = {
    "benchmarks/stt/results/model-metadata.json",
    "benchmarks/stt/results/model-metadata-turbo.json",
    "benchmarks/diarization/results/model-metadata.json",
}

# --- credential-value rules (conservative) ----------------------------------

PLACEHOLDER_VALUES = {
    "",
    "...",
    "…",
    "your-key",
    "YOUR_KEY",
    "REPLACE_ME",
    "changeme",
    "todo",
}

KEY_ASSIGNMENT_RE = re.compile(r"\bELEVENLABS_API_KEY\s*=\s*([^\s#'\"`]+)")
BEARER_RE = re.compile(r"Authorization\s*:\s*Bearer\s+([^\s'\"`]+)")
XI_API_KEY_RE = re.compile(r"xi-api-key\s*:\s*([^\s'\"`]+)")
LONG_ELEVENLABS_KEY_RE = re.compile(r"\bsk_[A-Za-z0-9]{20,}\b")

# Truncated fragments of an actual key would also be suspicious.
KEY_FRAGMENT_RE = re.compile(r"\bsk[-_][A-Za-z0-9]{24,}\b")

# printf/format placeholders used in documentation recipes (e.g. 'KEY=%s').
FORMAT_PLACEHOLDER_RE = re.compile(r"%[-#0-9.]*[a-zA-Z]")


def _normalize(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _is_placeholder(value: str) -> bool:
    # Trailing source escapes (\n, \t) are not part of the value.
    value = re.sub(r"\\[nrt]+$", "", value)
    if value in PLACEHOLDER_VALUES:
        return True
    # Environment expansion, code interpolation or documentation placeholder.
    if value.startswith(("${", "{", "<")):
        return True
    # Documentation recipes such as '%s' or '%s\n' are placeholders too.
    return bool(FORMAT_PLACEHOLDER_RE.match(value))


def path_violation(path: str) -> str | None:
    """Return a reason when the path itself must never be committed."""
    normalized = _normalize(path)
    name = Path(normalized).name
    suffix = Path(normalized).suffix.lower()

    if name.startswith(".env") and name not in ENV_ALLOWLIST:
        return "env secret file"

    if suffix in AUDIO_EXTENSIONS:
        return "audio file"

    if normalized.endswith(DATABASE_SUFFIXES):
        return "runtime database"

    if suffix in MODEL_EXTENSIONS:
        return "model binary"

    parts = Path(normalized).parts
    for directory in SENSITIVE_DIRECTORIES:
        if directory not in parts:
            continue
        if normalized in SAFE_RESULTS_FILES:
            continue
        if name == ".gitkeep":
            continue
        if directory == "results":
            return "benchmark output under results/"
        return f"private artifact under {directory}/"

    return None


def content_violation(path: str, text: str) -> str | None:
    """Return a reason when the file content contains a likely credential value."""
    for match in KEY_ASSIGNMENT_RE.finditer(text):
        if not _is_placeholder(match.group(1)):
            return "possible credential value in env assignment"
    for match in BEARER_RE.finditer(text):
        if not _is_placeholder(match.group(1)):
            return "possible credential value in Authorization header"
    for match in XI_API_KEY_RE.finditer(text):
        if not _is_placeholder(match.group(1)):
            return "possible credential value in xi-api-key header"
    if LONG_ELEVENLABS_KEY_RE.search(text) or KEY_FRAGMENT_RE.search(text):
        return "possible API key value"
    return None


def scan(entries: Iterable[tuple[str, str | None]]) -> list[tuple[str, str]]:
    """Scan (path, content) pairs; content may be None when it cannot be read as text."""
    violations: list[tuple[str, str]] = []
    for path, content in entries:
        reason = path_violation(path)
        if reason is not None:
            violations.append((path, reason))
            continue
        if content is not None:
            reason = content_violation(path, content)
            if reason is not None:
                violations.append((path, reason))
    return violations


def _git_lines(*args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def collect_files() -> list[str]:
    tracked = _git_lines("ls-files")
    staged = _git_lines("diff", "--cached", "--name-only", "--diff-filter=ACM")
    return sorted({_normalize(path) for path in (*tracked, *staged)})


def read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > 5 * 1024 * 1024:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def main() -> int:
    files = collect_files()
    entries = [(path, read_text(REPO_ROOT / path)) for path in files]
    violations = scan(entries)

    for path, reason in violations:
        print(f"{path}: {reason}")

    if violations:
        print(f"\nFAIL: {len(violations)} suspicious file(s) detected - do not commit.")
        return 1

    print(f"OK: {len(files)} tracked/staged files checked, no suspicious artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
