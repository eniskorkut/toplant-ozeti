"""Grounding, serialization and provider-guard tests (no external calls)."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.models import UNKNOWN_SPEAKER
from app.services.analysis_prompt import build_user_prompt, serialize_transcript
from app.services.analysis_schema import TranscriptTurnView, validate_payload
from app.services.llm_provider import (
    ELIGIBILITY_ERROR,
    LlmConfigurationError,
    build_provider,
    endpoint_is_restricted,
)

TURNS = [
    TranscriptTurnView(ordinal=0, speaker="Kişi 1", start_seconds=0.0, text="Merhaba"),
    TranscriptTurnView(ordinal=1, speaker="Kişi 2", start_seconds=5.42, text="Tamam"),
    TranscriptTurnView(ordinal=2, speaker=UNKNOWN_SPEAKER, start_seconds=9.81, text="???"),
]


def payload(**overrides) -> str:
    import json

    base = {
        "summary": "Toplantı özeti.",
        "topics": ["konu"],
        "decisions": [{"text": "Karar.", "source_turn_ordinals": [1]}],
        "action_items": [
            {
                "task": "Görev.",
                "owner": "Kişi 2",
                "due_date_text": None,
                "source_turn_ordinals": [1],
            }
        ],
        "important_moments": [
            {"title": "An", "description": "Açıklama", "source_turn_ordinal": 2}
        ],
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# --- serialization ---------------------------------------------------------


def test_serialization_is_deterministic_and_ordinal_based() -> None:
    text = serialize_transcript(TURNS)

    assert text == (
        "[0] 00:00.000 Kişi 1: Merhaba\n"
        "[1] 00:05.420 Kişi 2: Tamam\n"
        "[2] 00:09.810 Bilinmeyen: ???"
    )
    # deterministic: identical input, identical output
    assert serialize_transcript(TURNS) == text


def test_user_prompt_contains_only_transcript_lines() -> None:
    prompt = build_user_prompt(TURNS)
    assert "Transcript:" in prompt
    assert "/data/" not in prompt
    assert ".wav" not in prompt
    assert ".mp3" not in prompt


# --- validation ------------------------------------------------------------


def test_valid_payload_passes() -> None:
    parsed, errors = validate_payload(payload(), TURNS)
    assert errors == []
    assert parsed is not None
    assert parsed.decisions[0].source_turn_ordinals == [1]


def test_nonexistent_ordinal_is_rejected() -> None:
    _, errors = validate_payload(
        payload(decisions=[{"text": "Karar", "source_turn_ordinals": [99]}]), TURNS
    )
    assert any("does not exist" in error for error in errors)


def test_negative_ordinal_is_rejected() -> None:
    _, errors = validate_payload(
        payload(action_items=[{"task": "Görev", "source_turn_ordinals": [-1]}]), TURNS
    )
    assert any("negative ordinal" in error for error in errors)


def test_missing_source_turns_are_rejected() -> None:
    _, errors = validate_payload(
        payload(decisions=[{"text": "Karar", "source_turn_ordinals": []}]), TURNS
    )
    assert errors


def test_null_action_owner_is_accepted() -> None:
    parsed, errors = validate_payload(
        payload(action_items=[{"task": "Görev", "owner": None, "source_turn_ordinals": [0]}]),
        TURNS,
    )
    assert errors == []
    assert parsed is not None and parsed.action_items[0].owner is None


def test_existing_anonymous_owner_is_accepted() -> None:
    _, errors = validate_payload(
        payload(action_items=[{"task": "Görev", "owner": "Kişi 1", "source_turn_ordinals": [0]}]),
        TURNS,
    )
    assert errors == []


def test_nonexistent_owner_is_rejected() -> None:
    _, errors = validate_payload(
        payload(action_items=[{"task": "Görev", "owner": "Kişi 9", "source_turn_ordinals": [0]}]),
        TURNS,
    )
    assert any("unknown owner" in error for error in errors)


def test_unknown_speaker_cannot_be_action_owner() -> None:
    _, errors = validate_payload(
        payload(
            action_items=[
                {"task": "Görev", "owner": UNKNOWN_SPEAKER, "source_turn_ordinals": [0]}
            ]
        ),
        TURNS,
    )
    assert any("unresolved speech" in error for error in errors)


def test_empty_texts_are_rejected() -> None:
    _, errors = validate_payload(
        payload(decisions=[{"text": "", "source_turn_ordinals": [0]}]), TURNS
    )
    assert errors


def test_malformed_json_is_rejected() -> None:
    parsed, errors = validate_payload("{not json", TURNS)
    assert parsed is None
    assert errors


# --- provider eligibility guard --------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://opencode.ai/zen/go/v1",
        "https://opencode.ai/zen/go/v1/",
        "https://opencode.ai/zen/go/v1/chat/completions",
    ],
)
def test_opencode_go_endpoint_is_restricted(url: str) -> None:
    assert endpoint_is_restricted(url)


def test_other_endpoints_are_allowed() -> None:
    assert not endpoint_is_restricted("https://api.example.com/v1")


def test_restricted_provider_is_refused_and_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    monkeypatch.setattr(
        "httpx.post", lambda *args, **kwargs: calls.append(args) or pytest.fail("request sent")
    )
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="https://opencode.ai/zen/go/v1",
        llm_api_key="test-key-not-real",
        llm_model="test-model",
    )

    with pytest.raises(LlmConfigurationError) as excinfo:
        build_provider(settings)

    assert ELIGIBILITY_ERROR in str(excinfo.value)
    assert calls == []


def test_unconfigured_provider_fails_clearly() -> None:
    # Hermetic: never inherit MEETING_* values from the container environment.
    settings = Settings(
        llm_provider="openai_compatible", llm_base_url=None, llm_api_key=None, llm_model=None
    )
    with pytest.raises(LlmConfigurationError):
        build_provider(settings)
