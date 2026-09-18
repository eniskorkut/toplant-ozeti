"""Production merge: STT word timestamps + diarization segments.

Ported from the validated benchmark algorithm (benchmarks/merge/merge.py) with the
same rules. Application code does not import from benchmark directories.

Rules:
1. unique maximum temporal overlap wins;
2. equal maximum overlap -> midpoint if exactly one speaker owns it, else unresolved;
3. collapsed/zero-length word interval -> midpoint rule;
4. no overlap -> nearest segment within the boundary tolerance (<= 250 ms) only when
   exactly one speaker is nearest;
5. otherwise the word stays unresolved.

Unresolved words are preserved: they form their own turns carrying the explicit
unknown label, so text is never dropped and never falsely attributed.

Speaker labels are session-local: diarization cluster ids are mapped to "Kişi N" by
first appearance. Raw model cluster numbers are never exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models import UNKNOWN_SPEAKER

DEFAULT_BOUNDARY_TOLERANCE_SECONDS = 0.25
OVERLAP_EPSILON_SECONDS = 1e-9


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
class SpeakerTurn:
    speaker: str
    start: float
    end: float
    text: str = ""
    words: int = 0
    word_list: list[Word] = field(default_factory=list)


@dataclass
class MergeResult:
    turns: list[SpeakerTurn]
    assigned_words: int
    unresolved_words: int
    speaker_label_map: dict[str, str]

    @property
    def speaker_count(self) -> int:
        return len(self.speaker_label_map)


def overlap_seconds(word: Word, segment: DiarizationSegment) -> float:
    return min(word.end, segment.end) - max(word.start, segment.start)


def _midpoint_speaker(word: Word, segments: list[DiarizationSegment]) -> str | None:
    midpoint = (word.start + word.end) / 2.0
    owners = {segment.speaker for segment in segments if segment.start <= midpoint <= segment.end}
    return next(iter(owners)) if len(owners) == 1 else None


def assign_speaker(
    word: Word,
    segments: list[DiarizationSegment],
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE_SECONDS,
) -> str | None:
    overlaps = [(overlap_seconds(word, segment), segment) for segment in segments]
    max_overlap = max((overlap for overlap, _ in overlaps), default=0.0)

    if max_overlap > 0:
        top = [
            segment
            for overlap, segment in overlaps
            if overlap >= max_overlap - OVERLAP_EPSILON_SECONDS
        ]
        if len(top) == 1:
            return top[0].speaker
        return _midpoint_speaker(word, segments)

    midpoint_speaker = _midpoint_speaker(word, segments)
    if midpoint_speaker is not None:
        return midpoint_speaker

    candidates: list[tuple[float, str]] = []
    for segment in segments:
        gap = max(segment.start - word.end, word.start - segment.end)
        if gap <= tolerance:
            candidates.append((gap, segment.speaker))
    if not candidates:
        return None

    nearest_gap = min(gap for gap, _ in candidates)
    nearest_speakers = {
        speaker for gap, speaker in candidates if gap <= nearest_gap + OVERLAP_EPSILON_SECONDS
    }
    return next(iter(nearest_speakers)) if len(nearest_speakers) == 1 else None


def label_speakers(cluster_ids: list[str]) -> dict[str, str]:
    """Map anonymous cluster ids to session-local labels by first appearance."""
    labels: dict[str, str] = {}
    for cluster_id in cluster_ids:
        if cluster_id not in labels:
            labels[cluster_id] = f"Kişi {len(labels) + 1}"
    return labels


def merge_words(
    words: list[Word],
    segments: list[DiarizationSegment],
    *,
    tolerance: float = DEFAULT_BOUNDARY_TOLERANCE_SECONDS,
) -> MergeResult:
    """Assign speakers, form turns and produce session-local speaker labels."""
    cluster_sequence: list[str] = []
    for word in words:
        speaker = assign_speaker(word, segments, tolerance)
        cluster_sequence.append(speaker if speaker is not None else UNKNOWN_SPEAKER)
    label_map = label_speakers([c for c in cluster_sequence if c != UNKNOWN_SPEAKER])

    turns: list[SpeakerTurn] = []
    previous_speaker: str | None = None
    for word, cluster in zip(words, cluster_sequence, strict=True):
        speaker = label_map.get(cluster, UNKNOWN_SPEAKER)
        # Unresolved words are preserved as their own turns and terminate a run of
        # same-speaker words: text is kept, but never attributed to a neighbour.
        if turns and previous_speaker == speaker:
            turn = turns[-1]
            turn.end = word.end
            turn.text = f"{turn.text} {word.text}".strip()
            turn.words += 1
            turn.word_list.append(word)
        else:
            turns.append(
                SpeakerTurn(
                    speaker=speaker,
                    start=word.start,
                    end=word.end,
                    text=word.text,
                    words=1,
                    word_list=[word],
                )
            )
        previous_speaker = speaker

    assigned = sum(1 for cluster in cluster_sequence if cluster != UNKNOWN_SPEAKER)
    return MergeResult(
        turns=turns,
        assigned_words=assigned,
        unresolved_words=len(cluster_sequence) - assigned,
        speaker_label_map=label_map,
    )
