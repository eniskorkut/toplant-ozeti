#!/usr/bin/env python3
"""Create a UI-visible ElevenLabs comparison meeting for an existing recording.

Read-only reuse: the comparison meeting points at the source meeting's persisted MP3
and WAV paths; no audio is copied, mutated or re-uploaded. The source meeting row,
transcript and analysis are never modified (the script refuses to target the source id).

Run inside the backend image, e.g.:

    docker compose run --rm -e PYTHONPATH=/app \
        -v "$PWD/benchmarks/elevenlabs-stt/results/raw:/responses:ro" \
        -v "$PWD/scripts:/scripts:ro" \
        backend uv run --locked python /scripts/create_comparison_meeting.py \
            --source-meeting-id b1095740120b4b1e96db337da961ea63 \
            --comparison-id elevenlabs-thr022 \
            --response /responses/threshold-0.22.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")

from sqlalchemy import delete, select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.models import (  # noqa: E402
    MEETING_STATUS_COMPLETED,
    Meeting,
    TranscriptTurn,
)
from app.services.providers.elevenlabs import ElevenLabsProvider  # noqa: E402
from app.services.transcription import form_turns  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-meeting-id", required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--response", required=True, help="raw ElevenLabs response JSON")
    parser.add_argument("--provider-label", default="elevenlabs")
    parser.add_argument("--model-label", default="scribe_v2")
    args = parser.parse_args()

    if args.comparison_id == args.source_meeting_id:
        raise SystemExit("refusing to overwrite the source meeting")

    payload = json.loads(Path(args.response).read_text(encoding="utf-8"))
    words = ElevenLabsProvider.parse_words(payload)
    if not words:
        raise SystemExit("no words in the provided response")
    ElevenLabsProvider.validate_timestamps(words, audio_seconds=120.0)
    turns = form_turns(words)

    settings = get_settings()
    database = Database(settings.database_url)
    await database.init()

    async with database.session_factory() as session:
        source = await session.get(Meeting, args.source_meeting_id)
        if source is None:
            raise SystemExit("source meeting not found")
        source_audio = source.audio_mp3_path
        source_wav = source.processing_wav_path
        duration = source.duration_seconds
        if not (settings.meetings_dir / source_audio).is_file():
            raise SystemExit("source audio file missing")

        comparison = await session.get(Meeting, args.comparison_id)
        if comparison is None:
            comparison = Meeting(id=args.comparison_id, audio_mp3_path=source_audio,
                                 processing_wav_path=source_wav)
            session.add(comparison)

        comparison.status = MEETING_STATUS_COMPLETED
        comparison.duration_seconds = duration
        comparison.audio_mp3_path = source_audio  # read-only reuse, never duplicated
        comparison.processing_wav_path = source_wav
        comparison.requested_speaker_count = None
        comparison.processing_error = None
        comparison.transcription_provider = args.provider_label
        comparison.transcription_model = args.model_label
        comparison.requested_transcription_provider = args.provider_label

        await session.execute(
            delete(TranscriptTurn).where(TranscriptTurn.meeting_id == args.comparison_id)
        )
        for ordinal, turn in enumerate(turns):
            session.add(
                TranscriptTurn(
                    meeting_id=args.comparison_id,
                    ordinal=ordinal,
                    speaker=turn.speaker,
                    start_seconds=round(turn.start, 3),
                    end_seconds=round(turn.end, 3),
                    text=turn.text,
                )
            )
        await session.commit()

        source_after = await session.get(Meeting, args.source_meeting_id)
        print(
            json.dumps(
                {
                    "comparison_id": args.comparison_id,
                    "turns": len(turns),
                    "speakers": sorted({turn.speaker for turn in turns}),
                    "provider": comparison.transcription_provider,
                    "model": comparison.transcription_model,
                    "shares_source_audio": comparison.audio_mp3_path == source.audio_mp3_path,
                    "source_status_unchanged": source_after.status,
                    "source_provider_unchanged": source_after.transcription_provider,
                },
                ensure_ascii=False,
            )
        )
    await database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
