"""Benchmark-only merge: STT word timestamps + diarization segments.

Standard library only, so the algorithm can be unit-tested without any model or
container. Speaker labels stay anonymous; the diarization result is authoritative
and no short turn may be absorbed into a neighbouring speaker.

Rules implemented (as specified for this validation round):

1. assign each STT word the speaker with the largest temporal overlap;
2. if the word has a collapsed/zero-length interval, use its midpoint and pick the
   diarization segment active at that time;
3. if still unresolved, search only within a small boundary tolerance (default
   250 ms) around the word interval — never across a large gap;
4. otherwise the word is left unresolved (speaker=None) and counted, never assigned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_BOUNDARY_TOLERANCE_SECONDS = 0.25
SHORT_TURN_SECONDS = 1.0
RAPID_FLIP_SECONDS = 0.4


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    text: str = ""
    words: int = 0
    word_list: list[Word] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": self.words,
        }


def overlap_seconds(word: Word, segment: DiarizationSegment) -> float:
    return min(word.end, segment.end) - max(word.start, segment.start)


def assign_speaker(
    word: Word,
    segments: list[DiarizationSegment],
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE_SECONDS,
) -> str | None:
    """Return the diarization speaker for one word, or None when unresolved."""
    best_speaker: str | None = None
    best_overlap = 0.0
    for segment in segments:
        overlap = overlap_seconds(word, segment)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.speaker
    if best_speaker is not None:
        return best_speaker

    midpoint = (word.start + word.end) / 2.0
    for segment in segments:
        if segment.start <= midpoint <= segment.end:
            return segment.speaker

    nearest_speaker: str | None = None
    nearest_gap = tolerance
    for segment in segments:
        gap = max(segment.start - word.end, word.start - segment.end)
        if 0 <= gap < nearest_gap or (gap <= tolerance and gap < nearest_gap):
            nearest_gap = gap
            nearest_speaker = segment.speaker
    return nearest_speaker


def form_turns(words: list[Word], speakers: list[str | None]) -> list[Turn]:
    """Merge consecutive words of the same speaker; unresolved words break turns."""
    turns: list[Turn] = []
    for word, speaker in zip(words, speakers, strict=True):
        if speaker is None:
            continue
        if turns and turns[-1].speaker == speaker:
            turn = turns[-1]
            turn.end = word.end
            turn.text = f"{turn.text} {word.text}".strip()
            turn.words += 1
            turn.word_list.append(word)
        else:
            turns.append(
                Turn(
                    speaker=speaker,
                    start=word.start,
                    end=word.end,
                    text=word.text,
                    words=1,
                    word_list=[word],
                )
            )
    return turns


def structural_violations(
    words: list[Word],
    turns: list[Turn],
    audio_seconds: float,
) -> list[str]:
    """Return a list of invariant violations; empty means the merge is valid."""
    violations: list[str] = []

    previous_end = None
    for index, word in enumerate(words):
        if word.start < 0 or word.end < 0:
            violations.append(f"word {index} has a negative timestamp")
        if word.start > word.end:
            violations.append(f"word {index} has start > end")
        if previous_end is not None and word.start < previous_end - 1e-9:
            violations.append(f"word {index} breaks timestamp monotonicity")
        if word.end > audio_seconds + 0.01:
            violations.append(f"word {index} ends beyond the audio duration")
        previous_end = max(previous_end or 0.0, word.end)

    for index, turn in enumerate(turns):
        if turn.start > turn.end:
            violations.append(f"turn {index} has start > end")
        if not turn.text.strip():
            violations.append(f"turn {index} is empty")
        if index > 0 and turn.start < turns[index - 1].start - 1e-9:
            violations.append(f"turn {index} is not sorted by start time")
        if turn.end > audio_seconds + 0.01:
            violations.append(f"turn {index} ends beyond the audio duration")

    return violations


def timestamp_health(words: list[Word], audio_seconds: float) -> dict:
    """Timestamp usability indicators for one STT output."""
    zero_duration = 0
    overlaps = 0
    monotonicity_errors = 0
    impossible_gaps = 0
    out_of_range = 0
    previous_end = None

    for word in words:
        if abs(word.end - word.start) < 1e-9:
            zero_duration += 1
        if word.end < word.start:
            monotonicity_errors += 1
        if previous_end is not None and word.start < previous_end - 1e-9:
            monotonicity_errors += 1
        if previous_end is not None and word.start < previous_end:
            overlaps += 1
        if word.end > audio_seconds + 0.01 or word.start < -1e-9:
            out_of_range += 1
        if (word.end - word.start) > 30.0:
            impossible_gaps += 1
        previous_end = max(previous_end or 0.0, word.end)

    return {
        "words": len(words),
        "zero_duration_words": zero_duration,
        "monotonicity_errors": monotonicity_errors,
        "overlapping_word_timestamps": overlaps,
        "impossible_word_durations": impossible_gaps,
        "timestamps_out_of_range": out_of_range,
    }


def merge(
    words: list[Word],
    segments: list[DiarizationSegment],
    *,
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE_SECONDS,
    audio_seconds: float | None = None,
) -> dict:
    """Assign speakers, form turns and compute diagnostics."""
    speakers = [assign_speaker(word, segments, tolerance) for word in words]
    turns = form_turns(words, speakers)

    resolved = sum(1 for speaker in speakers if speaker is not None)
    unresolved = len(speakers) - resolved

    speaker_changes = sum(
        1 for index in range(1, len(turns)) if turns[index].speaker != turns[index - 1].speaker
    )
    short_turns = [
        turn
        for turn in turns
        if (turn.end - turn.start) < SHORT_TURN_SECONDS
    ]
    rapid_flips = 0
    for index in range(1, len(turns) - 1):
        previous, current, following = turns[index - 1], turns[index], turns[index + 1]
        if (
            current.speaker != previous.speaker
            and current.speaker != following.speaker
            and (current.end - current.start) < RAPID_FLIP_SECONDS
            and previous.speaker == following.speaker
        ):
            rapid_flips += 1

    violations = structural_violations(
        words, turns, audio_seconds if audio_seconds is not None else (words[-1].end if words else 0.0)
    )

    return {
        "words": len(words),
        "assignments": [
            {
                "start": word.start,
                "end": word.end,
                "text": word.text,
                "speaker": speaker,
            }
            for word, speaker in zip(words, speakers, strict=True)
        ],
        "assigned_words": resolved,
        "unresolved_words": unresolved,
        "turns": [turn.as_dict() for turn in turns],
        "speaker_changes": speaker_changes,
        "short_turns": len(short_turns),
        "short_turns_preserved": len(short_turns),
        "rapid_flips": rapid_flips,
        "structural_violations": violations,
        "per_speaker_words": {
            speaker: sum(1 for value in speakers if value == speaker)
            for speaker in sorted({value for value in speakers if value is not None})
        },
    }
