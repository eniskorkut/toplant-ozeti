"""Explicit real-model end-to-end test (deselected by default).

Run it deliberately, inside the backend image (which has whisper-cli + the mounted
models and the real recordings):

    docker compose exec backend uv run --locked pytest -m integration -q

The recording used is controlled by MEETING_INTEGRATION_RECORDING (default: the
45 s Turkish reference recording). No synthetic speech is used.
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

RECORDING_ID = os.environ.get("MEETING_INTEGRATION_RECORDING", "798b2efc586542d780eef3dce5dbbc8b")
MEETINGS_DIR = Path(os.environ.get("MEETING_DATA_DIR", "/data/meetings"))


def test_real_pipeline_produces_persisted_transcript(tmp_path: Path) -> None:
    recording_dir = MEETINGS_DIR / RECORDING_ID
    if not (recording_dir / "processing.wav").exists():
        pytest.skip(f"integration recording not available: {recording_dir}")

    settings = Settings(
        data_dir=MEETINGS_DIR,
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'integration.sqlite'}",
    )

    async def run() -> tuple[str, list[TranscriptTurn], float]:
        database = Database(settings.database_url)
        await database.init()
        async with database.session_factory() as session:
            session.add(
                Meeting(
                    id=RECORDING_ID,
                    status="queued",
                    audio_mp3_path=f"{RECORDING_ID}/meeting.mp3",
                    processing_wav_path=f"{RECORDING_ID}/processing.wav",
                )
            )
            await session.commit()

            meeting = await claim_next_meeting(session)
            assert meeting is not None
            await process_meeting(session, meeting, settings)

            refreshed = await session.get(Meeting, RECORDING_ID)
            turns = list(
                (
                    await session.execute(
                        select(TranscriptTurn)
                        .where(TranscriptTurn.meeting_id == RECORDING_ID)
                        .order_by(TranscriptTurn.ordinal)
                    )
                )
                .scalars()
                .all()
            )
        await database.dispose()
        return refreshed.status, turns, refreshed.duration_seconds or 0.0

    status, turns, duration = asyncio.run(run())

    assert status == MEETING_STATUS_COMPLETED
    assert turns, "transcript must not be empty"
    assert all(turn.text.strip() for turn in turns)
    assert all(0 <= turn.start_seconds <= turn.end_seconds for turn in turns)
    assert all(turn.end_seconds <= duration + 0.01 for turn in turns)
    assert all(turn.speaker.startswith("Kişi ") or turn.speaker == "Bilinmeyen" for turn in turns)

    with wave.open(str(recording_dir / "processing.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
