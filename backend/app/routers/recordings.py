import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_session
from app.models import MEETING_STATUS_UPLOADED, Meeting
from app.services.ffmpeg import FFmpegConversionError, FFmpegError, FFmpegNotFoundError
from app.services.recordings import (
    MP3_FILENAME,
    WAV_FILENAME,
    InvalidAudioFileError,
    UnsupportedAudioTypeError,
    process_recording,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


class RecordingInput(BaseModel):
    mime_type: str
    size_bytes: int


class RecordingMp3(BaseModel):
    size_bytes: int


class RecordingProcessingWav(BaseModel):
    size_bytes: int
    sample_rate: int
    channels: int


class RecordingCreated(BaseModel):
    recording_id: str
    meeting_id: str
    duration_seconds: float
    input: RecordingInput
    mp3: RecordingMp3
    processing_wav: RecordingProcessingWav
    conversion_ms: int


@router.post("", response_model=RecordingCreated, status_code=status.HTTP_201_CREATED)
async def create_recording(
    audio: Annotated[UploadFile, File(description="Audio recorded by the browser")],
    mime_type: Annotated[str, Form(description="MIME type reported by MediaRecorder")],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
    client_duration_seconds: Annotated[float | None, Form(ge=0)] = None,
) -> RecordingCreated:
    """Accept one recording, store it, and convert it to MP3 + 16 kHz mono WAV."""
    try:
        # Blocking file streaming + ffmpeg conversion run in a worker thread so the
        # API event loop is never blocked by the upload.
        artifacts = await asyncio.to_thread(
            process_recording,
            upload_file=audio.file,
            mime_type=mime_type,
            settings=settings,
        )
    except UnsupportedAudioTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except InvalidAudioFileError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except FFmpegConversionError as exc:
        logger.warning("ffmpeg could not convert the uploaded audio: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded audio could not be converted",
        ) from exc
    except (FFmpegError, FFmpegNotFoundError) as exc:
        logger.error("audio conversion failed on the server: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio conversion failed on the server",
        ) from exc
    finally:
        audio.file.close()

    if client_duration_seconds is not None:
        logger.info(
            "recording %s: client reported %.3fs, pipeline measured %.3fs",
            artifacts.recording_id,
            client_duration_seconds,
            artifacts.duration_seconds,
        )

    # The upload is also the meeting registration: the processing pipeline queues
    # meetings, never raw recordings.
    meeting = Meeting(
        id=artifacts.recording_id,
        status=MEETING_STATUS_UPLOADED,
        duration_seconds=round(artifacts.duration_seconds, 3),
        audio_mp3_path=f"{artifacts.recording_id}/{MP3_FILENAME}",
        processing_wav_path=f"{artifacts.recording_id}/{WAV_FILENAME}",
    )
    session.add(meeting)
    await session.commit()

    return RecordingCreated(
        recording_id=artifacts.recording_id,
        meeting_id=meeting.id,
        duration_seconds=round(artifacts.duration_seconds, 3),
        input=RecordingInput(
            mime_type=artifacts.input_mime_type,
            size_bytes=artifacts.input_size_bytes,
        ),
        mp3=RecordingMp3(size_bytes=artifacts.mp3_size_bytes),
        processing_wav=RecordingProcessingWav(
            size_bytes=artifacts.wav_size_bytes,
            sample_rate=artifacts.wav_sample_rate,
            channels=artifacts.wav_channels,
        ),
        conversion_ms=artifacts.conversion_ms,
    )
