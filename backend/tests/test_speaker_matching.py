"""Unit tests for cross-window speaker matching (no network, no provider)."""

from __future__ import annotations

from app.services.speaker_matching import (
    Interval,
    match_speakers,
    maximum_weight_matching,
    next_canonical_label,
    overlap_seconds,
    remap_final_speakers,
    score_pairs,
)


def test_overlap_seconds_basic_math() -> None:
    assert overlap_seconds((0.0, 2.0), (1.0, 3.0)) == 1.0
    assert overlap_seconds((0.0, 1.0), (2.0, 3.0)) == 0.0
    assert overlap_seconds((2.0, 3.0), (0.0, 4.0)) == 1.0


def test_request_local_speaker_ids_are_not_compared_by_name() -> None:
    # Window 1 says speaker_0 is on the left; window 2 reuses speaker_0 for the
    # other person in a different time region. Only overlap may decide identity.
    canonical = {"Kişi 1": [(0.0, 2.0)], "Kişi 2": [(2.0, 4.0)]}
    provider = {"speaker_0": [(2.1, 4.0)]}

    matches, unmatched = match_speakers(canonical, provider, window=(2.0, 4.0))

    assert unmatched == []
    assert matches["speaker_0"].canonical_speaker == "Kişi 2"


def test_low_confidence_overlap_creates_new_speaker_candidate() -> None:
    canonical = {"Kişi 1": [(0.0, 2.0)]}
    provider = {"speaker_0": [(1.95, 2.0)]}  # 50 ms of overlap: not enough

    matches, unmatched = match_speakers(canonical, provider, window=(1.9, 2.0))

    assert matches == {}
    assert unmatched == ["speaker_0"]


def test_one_to_one_matching_never_reuses_a_canonical() -> None:
    canonical = {"Kişi 1": [(0.0, 4.0)]}
    provider = {"speaker_0": [(0.0, 2.0)], "speaker_1": [(2.0, 4.0)]}

    matches, unmatched = match_speakers(canonical, provider, window=(0.0, 4.0))

    assert len(matches) == 1
    assert len(unmatched) == 1
    # The stronger overlap wins deterministically; both are equal here, so the
    # lower provider order wins.
    assert matches["speaker_0"].canonical_speaker == "Kişi 1"
    assert unmatched == ["speaker_1"]


def test_matching_is_deterministic_for_equal_scores() -> None:
    scores = {("Kişi 1", "speaker_a"): 1.0, ("Kişi 1", "speaker_b"): 1.0}
    first = maximum_weight_matching(scores)
    second = maximum_weight_matching(scores)
    assert first == second == {"speaker_a": "Kişi 1"}


def test_maximum_matching_beats_greedy_in_a_constructed_case() -> None:
    # Greedy on (Kişi 1, speaker_a)=5 would take speaker_a and leave Kişi 2 empty;
    # the maximum assignment is (Kişi 1, speaker_b) + (Kişi 2, speaker_a) = 9.
    scores = {
        ("Kişi 1", "speaker_a"): 5.0,
        ("Kişi 1", "speaker_b"): 4.0,
        ("Kişi 2", "speaker_a"): 5.0,
    }
    matching = maximum_weight_matching(scores)
    assert matching == {"speaker_a": "Kişi 2", "speaker_b": "Kişi 1"}


def test_score_pairs_clips_to_the_shared_window() -> None:
    canonical = {"Kişi 1": [(0.0, 10.0)]}
    provider = {"speaker_0": [(8.0, 10.0)]}
    scores = score_pairs(canonical, provider, window=(8.0, 10.0))
    assert scores[("Kişi 1", "speaker_0")] == 2.0


def test_next_canonical_label_skips_used_numbers() -> None:
    assert next_canonical_label(set()) == "Kişi 1"
    assert next_canonical_label({"Kişi 1", "Kişi 3"}) == "Kişi 2"


def test_remap_final_speakers_keeps_confident_labels_and_allocates_new_ones() -> None:
    live: dict[str, list[Interval]] = {
        "Kişi 1": [(0.0, 10.0), (20.0, 30.0)],
        "Kişi 2": [(10.0, 20.0)],
    }
    final = {
        "speaker_7": [(0.5, 10.0), (20.0, 29.0)],  # -> Kişi 1
        "speaker_2": [(10.0, 19.0)],  # -> Kişi 2
        "speaker_9": [(31.0, 40.0)],  # unseen: new label
    }

    mapping = remap_final_speakers(live, final)

    assert mapping["speaker_7"] == "Kişi 1"
    assert mapping["speaker_2"] == "Kişi 2"
    assert mapping["speaker_9"] == "Kişi 3"


def test_remap_never_applies_an_alias_to_an_unrelated_final_speaker() -> None:
    live: dict[str, list[Interval]] = {"Kişi 1": [(0.0, 5.0)]}
    final = {"speaker_x": [(100.0, 120.0)]}

    mapping = remap_final_speakers(live, final)

    assert mapping == {"speaker_x": "Kişi 2"}
