"""Usage endpoint tests (mocked HTTP, no network) and threshold plumbing guards."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.services.elevenlabs_usage import (
    REASON_NOT_CONFIGURED,
    REASON_REQUEST_FAILED,
    REASON_SCOPE_UNAVAILABLE,
    fetch_usage,
    normalize_subscription,
)


def settings(**overrides) -> Settings:
    values = {
        "elevenlabs_api_key": "sk_test_key_value",
        "llm_provider": "openai_compatible",
        "llm_base_url": None,
        "llm_api_key": None,
        "llm_model": None,
    }
    values.update(overrides)
    return Settings(**values)


class FakeResponse:
    def __init__(self, status_code: int, payload: object = None, *, bad_json: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._bad_json = bad_json

    def json(self) -> object:
        if self._bad_json:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def override(settings_obj: Settings) -> None:
    app.dependency_overrides[get_settings] = lambda: settings_obj


# --- normalization ----------------------------------------------------------


def test_normalizes_full_subscription_payload() -> None:
    usage = normalize_subscription(
        {
            "tier": "free",
            "status": "active",
            "character_count": 1200,
            "character_limit": 10000,
            "next_character_count_reset_unix": 1760000000,
        }
    )

    assert usage.available is True
    assert usage.tier == "free"
    assert usage.status == "active"
    assert usage.usage == 1200
    assert usage.limit == 10000
    assert usage.remaining == 8800
    assert usage.reset_at is not None and usage.reset_at.startswith("2025-")


def test_incomplete_payload_leaves_unknown_fields_null() -> None:
    usage = normalize_subscription({"tier": "creator"})

    assert usage.available is True
    assert usage.tier == "creator"
    assert usage.status is None
    assert usage.usage is None
    assert usage.limit is None
    assert usage.remaining is None
    assert usage.reset_at is None


def test_remaining_is_not_estimated_without_both_values() -> None:
    assert normalize_subscription({"character_count": 500}).remaining is None
    assert normalize_subscription({"character_limit": 500}).remaining is None


def test_malformed_payload_is_unavailable() -> None:
    assert normalize_subscription("not a dict").available is False
    assert normalize_subscription(None).reason == REASON_REQUEST_FAILED


# --- fetch behavior ---------------------------------------------------------


def test_fetch_usage_success(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_get(url, headers=None, timeout=None):
        captured.update({"url": url, "headers": headers})
        return FakeResponse(200, {"tier": "free", "character_count": 1, "character_limit": 10})

    monkeypatch.setattr("httpx.get", fake_get)
    usage = fetch_usage(settings())

    assert usage.available is True
    assert usage.remaining == 9
    assert captured["url"].endswith("/v1/user/subscription")
    assert set(captured["headers"]) == {"xi-api-key", "Accept"}


def test_fetch_usage_scope_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse(401))
    usage = fetch_usage(settings())

    assert usage.available is False
    assert usage.reason == REASON_SCOPE_UNAVAILABLE


def test_fetch_usage_forbidden_is_scope_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse(403))
    assert fetch_usage(settings()).reason == REASON_SCOPE_UNAVAILABLE


def test_fetch_usage_transport_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_get(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("httpx.get", failing_get)
    usage = fetch_usage(settings())

    assert usage.available is False
    assert usage.reason == REASON_REQUEST_FAILED


def test_fetch_usage_without_key_is_not_configured() -> None:
    usage = fetch_usage(settings(elevenlabs_api_key=None))
    assert usage.available is False
    assert usage.reason == REASON_NOT_CONFIGURED


def test_fetch_usage_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse(200, bad_json=True))
    assert fetch_usage(settings()).reason == REASON_REQUEST_FAILED


# --- endpoint ---------------------------------------------------------------


def test_usage_endpoint_reports_scope_unavailable_safely(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    override(settings())
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse(401))

    response = client.get("/api/v1/transcription/providers/elevenlabs/usage")
    body = response.json()

    assert response.status_code == 200
    assert body == {"available": False, "reason": REASON_SCOPE_UNAVAILABLE}


def test_usage_endpoint_success_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    override(settings())
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: FakeResponse(
            200,
            {
                "tier": "free",
                "status": "active",
                "character_count": 10,
                "character_limit": 100,
                "next_character_count_reset_unix": 1760000000,
            },
        ),
    )

    body = client.get("/api/v1/transcription/providers/elevenlabs/usage").json()

    assert body["available"] is True
    assert body["tier"] == "free"
    assert body["remaining"] == 90
    # Null fields are omitted from the response; nothing else is ever included.
    assert set(body) <= {
        "available",
        "tier",
        "status",
        "usage",
        "limit",
        "remaining",
        "reset_at",
        "reason",
    }
    assert "reason" not in body  # null fields are omitted


def test_usage_endpoint_never_contains_key_or_raw_payload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    override(settings(elevenlabs_api_key="sk_test_key_value"))
    monkeypatch.setattr(
        "httpx.get",
        lambda *args, **kwargs: FakeResponse(
            200,
            {
                "tier": "free",
                "character_count": 5,
                "character_limit": 50,
                "internal_note": "should never be exposed",
            },
        ),
    )

    text = client.get("/api/v1/transcription/providers/elevenlabs/usage").text

    assert "sk_test_key_value" not in text
    assert "internal_note" not in text
    assert "should never be exposed" not in text


def test_usage_lookup_failure_does_not_break_provider_listing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    override(settings())
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: FakeResponse(500))

    # A failing usage lookup must not affect provider capability reporting.
    providers = client.get("/api/v1/transcription/providers").json()
    usage = client.get("/api/v1/transcription/providers/elevenlabs/usage").json()

    assert providers["providers"][1]["available"] is True
    assert usage["available"] is False


# --- threshold request guards (no network) ---------------------------------


def test_threshold_only_sent_without_num_speakers() -> None:
    from app.services.providers.elevenlabs import ElevenLabsProvider

    automatic = ElevenLabsProvider.build_form_data(
        settings(elevenlabs_diarization_threshold=0.14), requested_speaker_count=None
    )
    known = ElevenLabsProvider.build_form_data(
        settings(elevenlabs_diarization_threshold=0.14), requested_speaker_count=4
    )

    assert automatic["diarization_threshold"] == "0.14"
    assert "num_speakers" not in automatic
    assert "diarization_threshold" not in known


def test_production_threshold_default_remains_null() -> None:
    assert settings().elevenlabs_diarization_threshold is None


def test_usage_response_has_no_json_payload_leak() -> None:
    # The normalized structure is a whitelist; prove it by construction.
    usage = normalize_subscription({"tier": "free", "extra": "leak"})
    assert "extra" not in json.dumps(usage.__dict__)
