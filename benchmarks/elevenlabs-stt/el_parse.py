"""Pure parsing/metrics for the ElevenLabs Scribe benchmark (standard library only).

Only entries with `type == "word"` count as words; spacing, audio events and other
metadata are ignored. A word without `speaker_id` stays unassigned — no speaker is ever
invented. Nothing here performs network calls, and no transcript text is written by the
aggregate helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

WORD_TYPE = "word"
UNKNOWN_SPEAKER = "Bilinmeyen"


@dataclass(frozen=True)
class ElevenWord:
    start: float
    end: float
    text: str
    speaker: str | None


@dataclass
class ElevenTurn:
    speaker: str
    start: float
    end: float
    text: str = ""
    words: int = 0
    word_list: list[ElevenWord] = field(default_factory=list)


class ElevenError(RuntimeError):
    """Base class for ElevenLabs benchmark failures."""


class ElevenConfigurationError(ElevenError):
    """Raised when the API key is missing."""


class ElevenQuotaError(ElevenError):
    """Raised when the API reports insufficient quota/plan for further calls."""


class ElevenRequestGuard:
    """Hard request budget: one requested benchmark call = one network call."""

    def __init__(self, maximum: int = 6) -> None:
        self.maximum = maximum
        self.used = 0
        self.audio_seconds_sent = 0.0
        self.calls: list[dict] = []

    def check(self) -> None:
        if self.used >= self.maximum:
            raise ElevenError(f"request budget exhausted ({self.maximum} calls)")

    def record(self, *, label: str, audio_seconds: float, latency_seconds: float, status: int) -> None:
        self.used += 1
        self.audio_seconds_sent += audio_seconds
        self.calls.append(
            {
                "label": label,
                "audio_seconds": round(audio_seconds, 3),
                "latency_seconds": round(latency_seconds, 3),
                "status": status,
            }
        )

    def summary(self) -> dict:
        return {
            "requests": self.used,
            "budget": self.maximum,
            "total_audio_seconds_sent": round(self.audio_seconds_sent, 3),
            "calls": self.calls,
        }


def redact_error(message: str, secrets: list[str]) -> str:
    """Remove secrets (e.g. the API key) from any error text before logging."""
    redacted = message
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted[:400]


def parse_words(payload: dict) -> list[ElevenWord]:
    """Extract spoken words only (type == 'word'), keeping missing speaker_ids absent."""
    words: list[ElevenWord] = []
    for entry in payload.get("words", []) or []:
        if entry.get("type") != WORD_TYPE:
            continue
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        speaker = entry.get("speaker_id")
        words.append(
            ElevenWord(
                start=float(entry.get("start", 0.0)),
                end=float(entry.get("end", 0.0)),
                text=text,
                speaker=str(speaker) if speaker not in (None, "") else None,
            )
        )
    return words


def group_turns(words: list[ElevenWord]) -> list[ElevenTurn]:
    """Group consecutive words of the same speaker; unassigned words keep their own turns."""
    turns: list[ElevenTurn] = []
    for word in words:
        label = word.speaker if word.speaker is not None else UNKNOWN_SPEAKER
        if turns and turns[-1].speaker == label:
            turn = turns[-1]
            turn.end = word.end
            turn.text = f"{turn.text} {word.text}".strip()
            turn.words += 1
            turn.word_list.append(word)
        else:
            turns.append(
                ElevenTurn(
                    speaker=label,
                    start=word.start,
                    end=word.end,
                    text=word.text,
                    words=1,
                    word_list=[word],
                )
            )
    return turns


def timestamp_health(words: list[ElevenWord]) -> dict:
    zero_duration = 0
    monotonicity = 0
    previous_end = None
    for word in words:
        if abs(word.end - word.start) < 1e-9:
            zero_duration += 1
        if word.end < word.start:
            monotonicity += 1
        if previous_end is not None and word.start < previous_end - 1e-9:
            monotonicity += 1
        previous_end = max(previous_end or 0.0, word.end)
    return {
        "zero_duration_words": zero_duration,
        "monotonicity_errors": monotonicity,
        "negative_timestamps": sum(1 for word in words if word.start < 0),
    }


def speaker_metrics(words: list[ElevenWord]) -> dict:
    turns = group_turns(words)
    tagged = [word for word in words if word.speaker is not None]
    untagged = [word for word in words if word.speaker is None]
    per_speaker: dict[str, int] = {}
    for word in tagged:
        assert word.speaker is not None  # narrowed by the filter above
        per_speaker[word.speaker] = per_speaker.get(word.speaker, 0) + 1

    labeled_turns = [turn for turn in turns if turn.speaker != UNKNOWN_SPEAKER]
    rapid = 0
    for index in range(1, len(turns) - 1):
        previous, current, following = turns[index - 1], turns[index], turns[index + 1]
        if (
            current.speaker not in (UNKNOWN_SPEAKER, previous.speaker, following.speaker)
            and (current.end - current.start) < 0.4
        ):
            rapid += 1

    return {
        "words": len(words),
        "speaker_tagged_words": len(tagged),
        "untagged_words": len(untagged),
        "distinct_speakers": len(per_speaker),
        "speaker_turns": len(labeled_turns),
        "unknown_turns": sum(1 for turn in turns if turn.speaker == UNKNOWN_SPEAKER),
        "rapid_flips": rapid,
        "words_per_speaker": dict(sorted(per_speaker.items())),
        "first_word_seconds": round(words[0].start, 3) if words else None,
        "last_word_seconds": round(words[-1].end, 3) if words else None,
        "timestamp_health": timestamp_health(words),
    }


def response_metrics(payload: dict) -> dict:
    """Safe summary of one response: counts and latency only, never transcript text."""
    words = parse_words(payload)
    text = "".join(word.text if index == 0 else f" {word.text}" for index, word in enumerate(words))
    return {
        "language_code": payload.get("language_code"),
        "language_probability": payload.get("language_probability"),
        "text_chars": len(text),
        **speaker_metrics(words),
    }
