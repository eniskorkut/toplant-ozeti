"""Strict schemas for grounded meeting analysis + semantic validation.

The model never produces timestamps: it references transcript turn ordinals and the
backend resolves them against persisted rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from app.models import UNKNOWN_SPEAKER


class Decision(BaseModel):
    text: str = Field(min_length=1)
    source_turn_ordinals: list[int] = Field(min_length=1)


class ActionItem(BaseModel):
    task: str = Field(min_length=1)
    owner: str | None = None
    due_date_text: str | None = None
    source_turn_ordinals: list[int] = Field(min_length=1)


class ImportantMoment(BaseModel):
    title: str = Field(min_length=1)
    description: str = ""
    source_turn_ordinal: int


class KeyPoint(BaseModel):
    text: str = Field(min_length=1)
    source_turn_ordinals: list[int] = Field(min_length=1)


class AnalysisPayload(BaseModel):
    summary: str = Field(min_length=1)
    key_points: list[KeyPoint] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    important_moments: list[ImportantMoment] = Field(default_factory=list)


class AnalysisValidationError(ValueError):
    """Raised when the payload violates the grounding rules."""


@dataclass(frozen=True)
class TranscriptTurnView:
    ordinal: int
    speaker: str
    start_seconds: float
    text: str
    # Meeting-scoped display alias for the canonical speaker, if the user set one.
    alias: str | None = None


def parse_payload(raw: str) -> AnalysisPayload:
    """Parse assistant content into the strict schema (Pydantic validation)."""
    return AnalysisPayload.model_validate_json(raw)


def semantic_errors(
    payload: AnalysisPayload, turns: list[TranscriptTurnView]
) -> list[str]:
    """Validate grounding against the persisted transcript turns."""
    errors: list[str] = []
    ordinals = {turn.ordinal for turn in turns}
    speakers = {turn.speaker for turn in turns}

    def check_ordinals(values: list[int], where: str) -> None:
        for ordinal in values:
            if ordinal < 0:
                errors.append(f"{where}: negative ordinal {ordinal}")
            elif ordinal not in ordinals:
                errors.append(f"{where}: ordinal {ordinal} does not exist")

    if not payload.summary.strip():
        errors.append("summary: empty")

    for index, key_point in enumerate(payload.key_points):
        if not key_point.text.strip():
            errors.append(f"key_points[{index}]: empty text")
        if not key_point.source_turn_ordinals:
            errors.append(f"key_points[{index}]: missing source turn ordinals")
        check_ordinals(key_point.source_turn_ordinals, f"key_points[{index}]")

    for index, decision in enumerate(payload.decisions):
        if not decision.text.strip():
            errors.append(f"decisions[{index}]: empty text")
        if not decision.source_turn_ordinals:
            errors.append(f"decisions[{index}]: missing source turn ordinals")
        check_ordinals(decision.source_turn_ordinals, f"decisions[{index}]")

    for index, item in enumerate(payload.action_items):
        if not item.task.strip():
            errors.append(f"action_items[{index}]: empty task")
        if not item.source_turn_ordinals:
            errors.append(f"action_items[{index}]: missing source turn ordinals")
        check_ordinals(item.source_turn_ordinals, f"action_items[{index}]")
        if item.owner is not None:
            if item.owner == UNKNOWN_SPEAKER:
                errors.append(
                    f"action_items[{index}]: '{UNKNOWN_SPEAKER}' is unresolved speech, not an owner"
                )
            elif item.owner not in speakers:
                errors.append(f"action_items[{index}]: unknown owner {item.owner!r}")

    for index, moment in enumerate(payload.important_moments):
        if not moment.title.strip():
            errors.append(f"important_moments[{index}]: empty title")
        check_ordinals([moment.source_turn_ordinal], f"important_moments[{index}]")

    return errors


def validate_payload(
    raw: str, turns: list[TranscriptTurnView]
) -> tuple[AnalysisPayload | None, list[str]]:
    """Return the parsed payload or the list of validation errors."""
    try:
        payload = parse_payload(raw)
    except ValidationError as exc:
        return None, [f"schema: {error['loc']} {error['msg']}" for error in exc.errors()][:10]

    errors = semantic_errors(payload, turns)
    if errors:
        return None, errors
    return payload, []
