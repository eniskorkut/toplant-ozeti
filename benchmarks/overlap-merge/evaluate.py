"""Pure evaluation helpers for the overlap-aware merge benchmark.

Agreement is measured only against reference RTTM turns (VoxConverse) using the best
cluster -> reference-speaker permutation. Wrong attribution is the share of compared
words whose mapped speaker disagrees with the reference at the word midpoint.
"""

from __future__ import annotations

from itertools import permutations

from context_resolver import Segment

UNKNOWN = "Bilinmeyen"


def parse_rttm(path) -> list[Segment]:
    turns: list[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        turns.append(Segment(start, start + float(parts[4]), parts[7]))
    return turns


def reference_speaker_at(turns: list[Segment], moment: float) -> str | None:
    for turn in turns:
        if turn.start <= moment <= turn.end:
            return turn.speaker
    return None


def best_permutation_mapping(
    words: list, speakers: list[str | None], reference: list[Segment]
) -> dict[str, str]:
    """Map anonymous clusters to reference speakers by maximum word agreement."""
    clusters = sorted({speaker for speaker in speakers if speaker is not None})
    reference_speakers = sorted({turn.speaker for turn in reference})
    limit = min(len(clusters), len(reference_speakers))
    best_mapping: dict[str, str] = {}
    best_matches = -1

    for permutation in permutations(reference_speakers, limit):
        mapping = {cluster: permutation[index] for index, cluster in enumerate(clusters[:limit])}
        matches = 0
        for word, speaker in zip(words, speakers, strict=True):
            if speaker is None or speaker not in mapping:
                continue
            moment = (word.start + word.end) / 2.0
            if reference_speaker_at(reference, moment) == mapping[speaker]:
                matches += 1
        if matches > best_matches:
            best_matches = matches
            best_mapping = mapping
    return best_mapping


def evaluate_assignment(
    words: list,
    speakers: list[str | None],
    reference: list[Segment],
    *,
    baseline_speakers: list[str | None] | None = None,
) -> dict:
    mapping = best_permutation_mapping(words, speakers, reference)

    compared = 0
    matches = 0
    wrong = 0
    for word, speaker in zip(words, speakers, strict=True):
        moment = (word.start + word.end) / 2.0
        reference_speaker = reference_speaker_at(reference, moment)
        if reference_speaker is None:
            continue
        if speaker is None:
            continue
        compared += 1
        if mapping.get(speaker) == reference_speaker:
            matches += 1
        else:
            wrong += 1

    unresolved = sum(1 for speaker in speakers if speaker is None)
    newly_resolved = 0
    if baseline_speakers is not None:
        newly_resolved = sum(
            1
            for index, speaker in enumerate(speakers)
            if speaker is not None and baseline_speakers[index] is None
        )

    return {
        "words": len(words),
        "compared_words": compared,
        "matching_words": matches,
        "wrong_attribution_words": wrong,
        "agreement": round(matches / compared, 4) if compared else None,
        "wrong_attribution_rate": round(wrong / compared, 4) if compared else None,
        "unresolved_words": unresolved,
        "unresolved_rate": round(unresolved / len(words), 4) if words else None,
        "newly_resolved_words": newly_resolved,
        "cluster_mapping": mapping,
    }


def speaker_metrics(words: list, speakers: list[str | None], unknown_label: str = UNKNOWN) -> dict:
    """Turn-level metrics derived from an assignment (no text needed)."""
    labeled = [speaker if speaker is not None else unknown_label for speaker in speakers]
    turns: list[dict] = []
    for word, speaker in zip(words, labeled, strict=True):
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["end"] = word.end
            turns[-1]["words"] += 1
        else:
            turns.append({"speaker": speaker, "start": word.start, "end": word.end, "words": 1})

    changes = sum(1 for index in range(1, len(turns)) if turns[index]["speaker"] != turns[index - 1]["speaker"])
    rapid = 0
    for index in range(1, len(turns) - 1):
        previous, current, following = turns[index - 1], turns[index], turns[index + 1]
        if (
            current["speaker"] != previous["speaker"]
            and current["speaker"] != following["speaker"]
            and (current["end"] - current["start"]) < 0.4
        ):
            rapid += 1

    kişi = {speaker for speaker in labeled if speaker.startswith("Kişi ")}
    return {
        "turns": len(turns),
        "speaker_changes": changes,
        "rapid_flips": rapid,
        "unknown_turns": sum(1 for turn in turns if turn["speaker"] == unknown_label),
        "kişi_count": len(kişi),
    }
