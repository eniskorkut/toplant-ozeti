"""Safety-aware selection rules for the diarization post-processing sweep.

False attribution is worse than leaving a word unresolved: candidates whose
wrong-attribution rate exceeds the production baseline by more than 0.005 absolute are
ineligible, regardless of how many unresolved words they recover.
"""

from __future__ import annotations

WRONG_ATTRIBUTION_TOLERANCE = 0.005


def is_eligible(baseline: dict, candidate: dict) -> bool:
    return (
        candidate["wrong_attribution_rate"]
        <= baseline["wrong_attribution_rate"] + WRONG_ATTRIBUTION_TOLERANCE
    )


def combined_assignment_error(entry: dict) -> float:
    total = entry["total_words"]
    if not total:
        return float("inf")
    return (entry["wrong_attribution_words"] + entry["unresolved_words"]) / total


def rank_key(entry: dict, *, current_on: float, current_off: float) -> tuple:
    return (
        combined_assignment_error(entry),
        entry["der"],
        entry["unresolved_rate"],
        entry["speaker_count_mae"],
        abs(entry["min_duration_on"] - current_on) + abs(entry["min_duration_off"] - current_off),
    )


def select_candidate(
    baseline: dict,
    candidates: list[dict],
    *,
    current_on: float = 0.3,
    current_off: float = 0.5,
) -> dict:
    """Return {"selected": entry | None, "eligible": [...], "reason": str}."""
    eligible = [entry for entry in candidates if is_eligible(baseline, entry)]
    if not eligible:
        return {
            "selected": None,
            "eligible": [],
            "reason": "no candidate stays within the wrong-attribution tolerance",
        }
    selected = min(eligible, key=lambda entry: rank_key(entry, current_on=current_on, current_off=current_off))
    return {
        "selected": selected,
        "eligible": [entry["config"] for entry in eligible],
        "reason": "lowest combined assignment error among eligible candidates",
    }
