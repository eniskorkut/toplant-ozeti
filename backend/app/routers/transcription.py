"""Transcription provider capabilities (safe metadata only, never secrets)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.config import Settings, get_settings

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
