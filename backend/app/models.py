"""Persistence models for the meeting processing pipeline.

Speaker labels stored in `TranscriptTurn.speaker` are session-local anonymous labels
("Kişi 1", "Kişi 2", …) or the explicit unknown label. No voiceprints, no embeddings
and no cross-meeting identity are persisted anywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

UNKNOWN_SPEAKER = "Bilinmeyen"

MEETING_STATUS_UPLOADED = "uploaded"
MEETING_STATUS_QUEUED = "queued"
MEETING_STATUS_PROCESSING = "processing"
MEETING_STATUS_COMPLETED = "completed"
MEETING_STATUS_FAILED = "failed"

MEETING_STATUSES = (
    MEETING_STATUS_UPLOADED,
    MEETING_STATUS_QUEUED,
    MEETING_STATUS_PROCESSING,
    MEETING_STATUS_COMPLETED,
    MEETING_STATUS_FAILED,
)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=MEETING_STATUS_UPLOADED, index=True)
    # Paths are stored relative to the meetings data directory (no absolute host paths).
    audio_mp3_path: Mapped[str] = mapped_column(String(255))
    processing_wav_path: Mapped[str] = mapped_column(String(255))
    requested_speaker_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    turns: Mapped[list[TranscriptTurn]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="TranscriptTurn.ordinal"
    )


class TranscriptTurn(Base):
    __tablename__ = "transcript_turns"
    __table_args__ = (Index("ix_transcript_turns_meeting_ordinal", "meeting_id", "ordinal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    speaker: Mapped[str] = mapped_column(String(32))
    start_seconds: Mapped[float] = mapped_column(Float)
    end_seconds: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)

    meeting: Mapped[Meeting] = relationship(back_populates="turns")
