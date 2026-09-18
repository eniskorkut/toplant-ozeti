"""Explicitly opt-in real ElevenLabs integration test (makes ONE cloud request).

Requirements:
    RUN_ELEVENLABS_INTEGRATION=1
    ELEVENLABS_API_KEY configured

It reuses the existing real four-speaker recording read-only, processes it through the
production worker/provider path into an isolated integration meeting and database, and
never touches the user's existing meeting row or transcript. Transcript content is not
printed — only counts, provider metadata and latency.
"""

from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import Settings
from app.db import Database
from app.models import MEETING_STATUS_COMPLETED, Meeting, TranscriptTurn
from app.services.pipeline import claim_next_meeting, process_meeting

pytestmark = pytest.mark.integration

REAL_MEETING = "b1095740120b4b1e96db337da961ea63"
MEETINGS_DIR = Path(os.environ.get("MEETING_DATA_DIR", "/data/meetings"))
INTEGRATION_MEETING = "integration-elevenlabs"


def test_real_elevenlabs_meeting_produces_speaker_turns(tmp_path: Path) -> None:
    if os.environ.get("RUN_ELEVENLABS_INTEGRATION") != "1":
        pytest.skip("set RUN_ELEVENLABS_INTEGRATION=1 to run the cloud integration test")
    if not os.environ.get("ELEVENLABS_API_KEY"):
        pytest.skip("ELEVENLABS_API_KEY is not configured")
    audio = MEETINGS_DIR / REAL_MEETING / "processing.wav"
    if not audio.is_file():
        pytest.skip(f"reference recording not available: {audio}")

    with wave.open(str(audio), "rb") as handle:
        audio_seconds = handle.getnframes() / handle.getframerate()

    settings = Settings(
        data_dir=MEETINGS_DIR,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'elevenlabs-integration.sqlite'}",
        transcription_provider="elevenlabs",
    )

    async def run() -> tuple[str, Meeting, list[TranscriptTurn]]:
        database = Database(settings.database_url)
        await database.init()
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=INTEGRATION_MEETING,
                    status="queued",
                    # Reuse the existing recording read-only under the integration id.
                    audio_mp3_path=f"{REAL_MEETING}/meeting.mp3",
                    processing_wav_path=f"{REAL_MEETING}/processing.wav",
                )
            )
            await session.commit()
            meeting = await claim_next_meeting(session)
            assert meeting is not None
            await process_meeting(session, meeting, settings)
            refreshed = await session.get(Meeting, INTEGRATION_MEETING)
            turns = list(
                (
                    await session.execute(
                        select(TranscriptTurn)
                        .where(TranscriptTurn.meeting_id == INTEGRATION_MEETING)
                        .order_by(TranscriptTurn.ordinal)
                    )
                )
                .scalars()
                .all()
            )
        await database.dispose()
        return refreshed.status, refreshed, turns

    status, meeting, turns = asyncio.run(run())

    speakers = sorted({turn.speaker for turn in turns if turn.speaker != "Bilinmeyen"})
    untagged = sum(1 for turn in turns if turn.speaker == "Bilinmeyen")
    print(
        "elevenlabs integration summary:",
        {
            "status": status,
            "provider": meeting.transcription_provider,
            "model": meeting.transcription_model,
            "audio_seconds": round(audio_seconds, 2),
            "words": sum(len(turn.text.split()) for turn in turns),
            "speakers": speakers,
            "untagged_turns": untagged,
            "turns": len(turns),
        },
        flush=True,
    )

    assert status == MEETING_STATUS_COMPLETED
    assert meeting.transcription_provider == "elevenlabs"
    assert meeting.transcription_model
    assert turns
