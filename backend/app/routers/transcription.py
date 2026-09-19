"""Transcription provider capabilities (safe metadata only, never secrets)."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.services.elevenlabs_usage import fetch_usage
from app.services.providers.elevenlabs import create_realtime_token
from app.services.transcription import (
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderUnavailableError,
)

router = APIRouter(prefix="/api/v1/transcription", tags=["transcription"])

PROVIDER_LABELS = {
    "local": "Yerel",
    "elevenlabs": "ElevenLabs",
}


class ProviderCapability(BaseModel):
    id: str
    available: bool
    cloud: bool
    label: str


class ProviderCapabilities(BaseModel):
    default: str
    providers: list[ProviderCapability]


@router.get("/providers", response_model=ProviderCapabilities)
async def list_providers(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProviderCapabilities:
    """Availability is derived from configuration only; no key material is returned."""
    return ProviderCapabilities(
        default=settings.transcription_provider,
        providers=[
            ProviderCapability(
                id="local",
                available=True,
                cloud=False,
                label=PROVIDER_LABELS["local"],
            ),
            ProviderCapability(
                id="elevenlabs",
                available=settings.elevenlabs_configured,
                cloud=True,
                label=PROVIDER_LABELS["elevenlabs"],
            ),
        ],
    )


class ElevenLabsUsageResponse(BaseModel):
    available: bool
    tier: str | None = None
    status: str | None = None
    usage: int | None = None
    limit: int | None = None
    remaining: int | None = None
    reset_at: str | None = None
    reason: str | None = None


class RealtimeTokenResponse(BaseModel):
    token: str


@router.post("/providers/elevenlabs/realtime-token", response_model=RealtimeTokenResponse)
async def elevenlabs_realtime_token(
    settings: Annotated[Settings, Depends(get_settings)],
) -> RealtimeTokenResponse:
    """Mint a short-lived single-use token for the browser realtime client.

    Only the single-use token is returned; the permanent API key stays server-side,
    is never logged and is never included in errors.
    """
    if not settings.elevenlabs_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs transcription is not configured.",
        )
    try:
        token = await asyncio.to_thread(create_realtime_token, settings)
    except ProviderConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ElevenLabs is temporarily unavailable.",
        ) from exc
    except (ProviderRequestError, ProviderResponseError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ElevenLabs rejected the realtime token request.",
        ) from exc
    return RealtimeTokenResponse(token=token)


@router.get(
    "/providers/elevenlabs/usage",
    response_model=ElevenLabsUsageResponse,
    response_model_exclude_none=True,
)
async def elevenlabs_usage(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ElevenLabsUsageResponse:
    """Safe usage summary; never returns keys, headers or raw provider payloads.

    Exact remaining characters are only reported when the provider returns both the
    usage and the limit; remaining time is never estimated from character counts.
    """
    usage = fetch_usage(settings)
    return ElevenLabsUsageResponse(
        available=usage.available,
        tier=usage.tier,
        status=usage.status,
        usage=usage.usage,
        limit=usage.limit,
        remaining=usage.remaining,
        reset_at=usage.reset_at,
        reason=usage.reason,
    )
