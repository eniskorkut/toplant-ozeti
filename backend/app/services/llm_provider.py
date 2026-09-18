"""Provider-agnostic meeting-analysis LLM interface.

Only OpenAI-compatible chat completions are implemented (Bearer auth, JSON-only
content parsed locally). No provider URL or model is hard-coded in business logic and
no secret is ever logged or returned.

A temporary eligibility guard refuses to send meeting content to endpoints that are
documented for other kinds of traffic (currently OpenCode Go / coding-agent traffic).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

ELIGIBILITY_ERROR = (
    "Configured LLM endpoint is restricted to coding-agent traffic and is not enabled "
    "for meeting analysis."
)

RESTRICTED_ENDPOINTS = (
    "opencode.ai/zen/go/v1",
)


class LlmProviderError(RuntimeError):
    """Base class for provider failures."""


class LlmConfigurationError(LlmProviderError):
    """Raised when the provider is not configured or not eligible."""


class LlmTransportError(LlmProviderError):
    """Connection or timeout failure (retryable)."""


class LlmHttpError(LlmProviderError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.status_code == 429 or self.status_code >= 500


class LlmResponseError(LlmProviderError):
    """Malformed envelope, missing choices or empty content (not retryable here)."""


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    provider: str
    model: str
    latency_seconds: float


class MeetingAnalysisProvider(Protocol):
    def analyze(self, *, system_prompt: str, user_prompt: str, session_id: str) -> ProviderResponse:
        ...


def endpoint_is_restricted(base_url: str) -> bool:
    """Temporary guard: refuse endpoints reserved for coding-agent traffic."""
    normalized = base_url.strip().rstrip("/").lower()
    parsed = urlparse(normalized if "://" in normalized else f"https://{normalized}")
    candidate = f"{parsed.netloc}{parsed.path}".rstrip("/")
    return any(candidate.startswith(restricted) for restricted in RESTRICTED_ENDPOINTS)


def build_provider(settings: Settings) -> MeetingAnalysisProvider:
    if settings.llm_provider == "mock":
        return MockProvider()
    if not settings.llm_configured:
        raise LlmConfigurationError(
            "No meeting-analysis LLM provider is configured "
            "(MEETING_LLM_BASE_URL, MEETING_LLM_API_KEY, MEETING_LLM_MODEL)."
        )
    assert settings.llm_base_url is not None  # narrowed by llm_configured
    if endpoint_is_restricted(settings.llm_base_url):
        raise LlmConfigurationError(ELIGIBILITY_ERROR)
    return OpenAICompatibleProvider(settings)


class OpenAICompatibleProvider:
    """POST {base_url}/chat/completions with Bearer authentication."""

    def __init__(self, settings: Settings) -> None:
        assert settings.llm_base_url and settings.llm_api_key and settings.llm_model
        self._base_url = settings.llm_base_url.rstrip("/")
        self._api_key = settings.llm_api_key.get_secret_value()
        self._model = settings.llm_model
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries

    @property
    def provider_name(self) -> str:
        return urlparse(self._base_url).netloc or "openai-compatible"

    def analyze(self, *, system_prompt: str, user_prompt: str, session_id: str) -> ProviderResponse:
        if endpoint_is_restricted(self._base_url):
            raise LlmConfigurationError(ELIGIBILITY_ERROR)

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-Session-Id": session_id,
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = LlmTransportError(f"provider connection failed: {type(exc).__name__}")
                logger.warning("analysis provider transport error (attempt %d)", attempt + 1)
                self._backoff(attempt)
                continue

            latency = time.perf_counter() - started
            if response.status_code >= 400:
                error = LlmHttpError(
                    response.status_code, f"provider returned HTTP {response.status_code}"
                )
                if not error.retryable:
                    # Never retry auth/forbidden or other client errors.
                    raise error
                last_error = error
                logger.warning(
                    "analysis provider HTTP %d (attempt %d)", response.status_code, attempt + 1
                )
                self._backoff(attempt)
                continue

            return ProviderResponse(
                content=self._extract_content(response),
                provider=self.provider_name,
                model=self._model,
                latency_seconds=round(latency, 3),
            )

        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_content(response: httpx.Response) -> str:
        try:
            envelope = response.json()
        except ValueError as exc:
            raise LlmResponseError("provider envelope was not valid JSON") from exc
        if not isinstance(envelope, dict):
            raise LlmResponseError("provider envelope was not an object")
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LlmResponseError("provider response contained no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LlmResponseError("provider returned empty assistant content")
        return content

    def _backoff(self, attempt: int) -> None:
        if attempt < self._max_retries:
            time.sleep(min(2.0, 0.5 * (2**attempt)))


class MockProvider:
    """Deterministic provider for tests and the local mock E2E (never external)."""

    def __init__(self, content: str | None = None) -> None:
        self._content = content
        self.calls = 0

    def analyze(
        self, *, system_prompt: str, user_prompt: str, session_id: str
    ) -> ProviderResponse:
        self.calls += 1
        content = self._content if self._content is not None else json.dumps(
            {
                "summary": "Mock analysis summary.",
                "topics": ["mock topic"],
                "decisions": [{"text": "Mock decision.", "source_turn_ordinals": [0]}],
                "action_items": [
                    {
                        "task": "Mock task.",
                        "owner": None,
                        "due_date_text": None,
                        "source_turn_ordinals": [0],
                    }
                ],
                "important_moments": [
                    {
                        "title": "Mock moment",
                        "description": "Mock description.",
                        "source_turn_ordinal": 0,
                    }
                ],
            }
        )
        return ProviderResponse(
            content=content,
            provider="mock",
            model="mock-model",
            latency_seconds=0.0,
        )
