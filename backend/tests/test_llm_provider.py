"""HTTP failure-policy tests for the OpenAI-compatible provider (fake transport)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.config import Settings
from app.services.llm_provider import (
    LlmHttpError,
    LlmResponseError,
    LlmTransportError,
    OpenAICompatibleProvider,
)

VALID_CONTENT = json.dumps({"summary": "ok"})


def settings(**overrides) -> Settings:
    values = {
        "llm_base_url": "https://api.example.com/v1",
        "llm_api_key": "not-a-real-key",
        "llm_model": "test-model",
        "llm_fallback_model": None,
        "llm_fallback_models": [],
        "llm_timeout_seconds": 5.0,
        "llm_max_retries": 2,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


class FakeResponse:
    def __init__(self, status_code: int, payload=None, *, json_error: bool = False) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("not json")
        return self._payload


def envelope(content: str | None = None, *, choices=None) -> dict:
    if choices is not None:
        return {"choices": choices}
    return {
        "choices": [{"message": {"content": content if content is not None else VALID_CONTENT}}]
    }


def run_with(monkeypatch: pytest.MonkeyPatch, responses: list) -> tuple[object, int]:
    calls = {"count": 0}

    def fake_post(*args, **kwargs):
        index = min(calls["count"], len(responses) - 1)
        result = responses[index]
        calls["count"] += 1
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr("httpx.post", fake_post)
    provider = OpenAICompatibleProvider(settings())
    outcome = None
    error = None
    try:
        outcome = provider.analyze(system_prompt="s", user_prompt="u", session_id="m1")
    except Exception as exc:  # noqa: BLE001 - the test asserts on the error type
        error = exc
    return (outcome, error), calls["count"]


def test_valid_response(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), calls = run_with(monkeypatch, [FakeResponse(200, envelope())])
    assert error is None
    assert outcome is not None and outcome.content == VALID_CONTENT
    assert calls == 1


def test_timeout_is_retried_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), calls = run_with(
        monkeypatch, [httpx.TimeoutException("timeout"), httpx.TimeoutException("timeout")]
    )
    assert isinstance(error, LlmTransportError)
    assert calls == 3  # initial + 2 retries


def test_429_is_retried_and_can_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), calls = run_with(
        monkeypatch, [FakeResponse(429), FakeResponse(200, envelope())]
    )
    assert error is None
    assert outcome is not None
    assert calls == 2


def test_5xx_is_retried_then_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), calls = run_with(
        monkeypatch, [FakeResponse(503), FakeResponse(503), FakeResponse(503)]
    )
    assert isinstance(error, LlmHttpError) and error.status_code == 503
    assert calls == 3


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    (outcome, error), calls = run_with(monkeypatch, [FakeResponse(status)])
    assert isinstance(error, LlmHttpError) and error.status_code == status
    assert calls == 1


def test_missing_choices_fails_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), calls = run_with(monkeypatch, [FakeResponse(200, {})])
    assert isinstance(error, LlmResponseError)
    assert calls == 1


def test_empty_content_fails_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), _ = run_with(monkeypatch, [FakeResponse(200, envelope("   "))])
    assert isinstance(error, LlmResponseError)


def test_malformed_envelope_fails_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    (outcome, error), _ = run_with(monkeypatch, [FakeResponse(200, json_error=True)])
    assert isinstance(error, LlmResponseError)


def test_provider_never_logs_or_returns_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging

    captured: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            captured.append(message % record.args if record.args else message)

    handler = Collector()
    logging.getLogger("app.services.llm_provider").addHandler(handler)
    try:
        run_with(monkeypatch, [FakeResponse(500), FakeResponse(500), FakeResponse(500)])
    finally:
        logging.getLogger("app.services.llm_provider").removeHandler(handler)

    assert all("not-a-real-key" not in message for message in captured)


def test_x_opencode_session_header_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_headers: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured_headers.update(headers or {})
        return FakeResponse(200, envelope())

    monkeypatch.setattr("httpx.post", fake_post)
    provider = OpenAICompatibleProvider(settings())
    provider.analyze(system_prompt="s", user_prompt="u", session_id="session-xyz")
    assert captured_headers.get("x-opencode-session") == "session-xyz"
    assert captured_headers.get("X-Session-Id") == "session-xyz"


def test_fallback_model_used_when_primary_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads_sent: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        payloads_sent.append(json or {})
        if json and json.get("model") == "model-primary":
            return FakeResponse(401, {"error": {"message": "Model model-primary is not supported"}})
        return FakeResponse(200, envelope())

    monkeypatch.setattr("httpx.post", fake_post)
    s = settings(llm_model="model-primary", llm_fallback_model="model-fallback")
    provider = OpenAICompatibleProvider(s)
    res = provider.analyze(system_prompt="s", user_prompt="u", session_id="m1")

    assert res.model == "model-fallback"
    assert len(payloads_sent) == 2
    assert payloads_sent[0]["model"] == "model-primary"
    assert payloads_sent[1]["model"] == "model-fallback"


def test_opencode_go_aliases_flash_free() -> None:
    s = settings(
        llm_base_url="https://opencode.ai/zen/go/v1",
        llm_model="deepseek-v4-flash-free",
        llm_fallback_model="deepseek-v4.1-flash",
    )
    provider = OpenAICompatibleProvider(s)
    assert provider.models == ["deepseek-v4-flash", "deepseek-v4.1-flash"]
