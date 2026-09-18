"""ElevenLabs subscription/usage lookup, normalized to safe non-secret fields.

The account key used here may be scoped without billing access; that is reported as
`usage_scope_unavailable` instead of failing transcription. Only explicitly returned
fields are exposed — remaining is exact arithmetic on provider values and is never
converted into time estimates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

SUBSCRIPTION_URL = "https://api.elevenlabs.io/v1/user/subscription"

REASON_NOT_CONFIGURED = "not_configured"
REASON_SCOPE_UNAVAILABLE = "usage_scope_unavailable"
REASON_REQUEST_FAILED = "usage_request_failed"


@dataclass(frozen=True)
class ElevenLabsUsage:
    available: bool
    tier: str | None = None
    status: str | None = None
    usage: int | None = None
    limit: int | None = None
    remaining: int | None = None
    reset_at: str | None = None
    reason: str | None = None


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _reset_iso(value: object) -> str | None:
    seconds = _as_int(value)
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=UTC).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def normalize_subscription(payload: object) -> ElevenLabsUsage:
    """Whitelist the meaningful provider fields; never pass the payload through."""
    if not isinstance(payload, dict):
        return ElevenLabsUsage(available=False, reason=REASON_REQUEST_FAILED)

    usage = _as_int(payload.get("character_count"))
    limit = _as_int(payload.get("character_limit"))
    remaining = limit - usage if usage is not None and limit is not None else None

    tier = payload.get("tier")
    status = payload.get("status")

    return ElevenLabsUsage(
        available=True,
        tier=str(tier) if isinstance(tier, str) else None,
        status=str(status) if isinstance(status, str) else None,
        usage=usage,
        limit=limit,
        remaining=remaining,
        reset_at=_reset_iso(payload.get("next_character_count_reset_unix")),
    )


def fetch_usage(settings: Settings) -> ElevenLabsUsage:
    secret = settings.elevenlabs_api_key
    api_key = secret.get_secret_value() if secret is not None else ""
    if not api_key.strip():
        return ElevenLabsUsage(available=False, reason=REASON_NOT_CONFIGURED)

    try:
        response = httpx.get(
            SUBSCRIPTION_URL,
            headers={"xi-api-key": api_key, "Accept": "application/json"},
            timeout=settings.elevenlabs_timeout_seconds,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        logger.warning("elevenlabs usage lookup failed: %s", type(exc).__name__)
        return ElevenLabsUsage(available=False, reason=REASON_REQUEST_FAILED)

    if response.status_code in (401, 403):
        # A scoped key without billing access is expected; transcription still works.
        logger.info("elevenlabs usage endpoint is not accessible with this key scope")
        return ElevenLabsUsage(available=False, reason=REASON_SCOPE_UNAVAILABLE)
    if response.status_code != 200:
        return ElevenLabsUsage(available=False, reason=REASON_REQUEST_FAILED)

    try:
        payload = response.json()
    except ValueError:
        return ElevenLabsUsage(available=False, reason=REASON_REQUEST_FAILED)

    return normalize_subscription(payload)
