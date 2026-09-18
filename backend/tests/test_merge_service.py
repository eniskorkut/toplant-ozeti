"""Unit tests for the production merge rules (no ML models)."""

from __future__ import annotations

from app.models import UNKNOWN_SPEAKER
from app.services.merge import (
    DiarizationSegment,
    Word,
    assign_speaker,
    label_speakers,
    merge_words,
)

SEGMENTS = [
    DiarizationSegment(0.0, 2.0, "speaker_0"),
    DiarizationSegment(2.0, 4.0, "speaker_1"),
]


def test_unique_max_overlap_wins() -> None:
    assert assign_speaker(Word(1.8, 2.6, "x"), SEGMENTS) == "speaker_1"


def test_equal_overlap_on_shared_boundary_is_unresolved() -> None:
    segments = [DiarizationSegment(0.0, 1.0, "a"), DiarizationSegment(1.0, 3.0, "b")]
    assert assign_speaker(Word(0.0, 2.0, "x"), segments) is None


def test_equal_overlap_ambiguity_is_order_independent() -> None:
    segments = [DiarizationSegment(0.0, 1.0, "a"), DiarizationSegment(1.0, 3.0, "b")]
    word = Word(0.0, 2.0, "x")
    assert assign_speaker(word, segments) == assign_speaker(word, list(reversed(segments)))


def test_collapsed_interval_uses_midpoint() -> None:
    assert assign_speaker(Word(1.6, 1.6, "x"), SEGMENTS) == "speaker_0"


def test_small_gap_within_tolerance() -> None:
    assert assign_speaker(Word(4.1, 4.2, "x"), SEGMENTS, tolerance=0.25) == "speaker_1"


def test_gap_beyond_tolerance_is_unresolved() -> None:
    assert assign_speaker(Word(4.5, 4.6, "x"), SEGMENTS, tolerance=0.25) is None


def test_equidistant_tolerance_candidates_are_unresolved() -> None:
    segments = [DiarizationSegment(0.0, 1.0, "a"), DiarizationSegment(2.0, 3.0, "b")]
    assert assign_speaker(Word(1.4, 1.6, "x"), segments, tolerance=0.25) is None


def test_unresolved_word_breaks_same_speaker_turn() -> None:
    words = [Word(0.0, 0.5, "bir"), Word(5.0, 5.5, "belirsiz"), Word(6.0, 6.5, "iki")]
    segments = [
        DiarizationSegment(0.0, 1.0, "speaker_0"),
        DiarizationSegment(6.0, 7.0, "speaker_0"),
    ]

    result = merge_words(words, segments)

    assert [turn.speaker for turn in result.turns] == ["Kişi 1", UNKNOWN_SPEAKER, "Kişi 1"]
    assert result.unresolved_words == 1


def test_unresolved_text_is_preserved_with_unknown_label() -> None:
    words = [Word(0.0, 0.5, "a"), Word(5.0, 5.5, "kayip")]
    result = merge_words(words, [DiarizationSegment(0.0, 1.0, "speaker_0")])

    unknown_turns = [turn for turn in result.turns if turn.speaker == UNKNOWN_SPEAKER]
    assert len(unknown_turns) == 1
    assert unknown_turns[0].text == "kayip"
    assert result.unresolved_words == 1


def test_short_turn_is_preserved() -> None:
    words = [Word(0.0, 0.5, "Evet"), Word(2.0, 2.2, "devam")]
    segments = [
        DiarizationSegment(0.0, 1.0, "speaker_0"),
        DiarizationSegment(1.9, 3.0, "speaker_1"),
    ]
    result = merge_words(words, segments)

    assert len(result.turns) == 2
    assert result.turns[0].text == "Evet"
    assert result.turns[1].speaker == "Kişi 2"


def test_speaker_labels_follow_first_appearance() -> None:
    # cluster_7 appears first, cluster_2 second -> Kişi 1 / Kişi 2
    words = [Word(0.0, 0.5, "a"), Word(1.0, 1.5, "b"), Word(2.0, 2.5, "c")]
    segments = [
        DiarizationSegment(0.0, 0.6, "cluster_7"),
        DiarizationSegment(0.9, 1.6, "cluster_2"),
        DiarizationSegment(1.9, 2.6, "cluster_7"),
    ]
    result = merge_words(words, segments)

    assert result.speaker_label_map == {"cluster_7": "Kişi 1", "cluster_2": "Kişi 2"}
    assert [turn.speaker for turn in result.turns] == ["Kişi 1", "Kişi 2", "Kişi 1"]


def test_label_speakers_is_deterministic() -> None:
    assert label_speakers(["c", "a", "c", "b"]) == {"c": "Kişi 1", "a": "Kişi 2", "b": "Kişi 3"}


def test_turns_are_sorted_and_non_overlapping_in_start_time() -> None:
    words = [Word(0.0, 0.4, "a"), Word(0.5, 0.9, "b"), Word(3.0, 3.4, "c")]
    segments = [DiarizationSegment(0.0, 1.0, "s0"), DiarizationSegment(2.9, 4.0, "s1")]
    result = merge_words(words, segments)
    starts = [turn.start for turn in result.turns]
    assert starts == sorted(starts)
