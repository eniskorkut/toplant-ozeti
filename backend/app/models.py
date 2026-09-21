"""Persistence models for the meeting processing pipeline.

Speaker labels stored in `TranscriptTurn.speaker` are session-local anonymous labels
("Kişi 1", "Kişi 2", …) or the explicit unknown label. No voiceprints, no embeddings
and no cross-meeting identity are persisted anywhere.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
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

# Analysis lifecycle is independent of the transcript lifecycle.
ANALYSIS_STATUS_QUEUED = "queued"
ANALYSIS_STATUS_PROCESSING = "processing"
ANALYSIS_STATUS_COMPLETED = "completed"
ANALYSIS_STATUS_FAILED = "failed"


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
    # How the transcript was produced (safe metadata only; never keys or raw responses).
    transcription_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    transcription_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # What the client asked for; null means "use the configured server default".
    requested_transcription_provider: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )

    turns: Mapped[list[TranscriptTurn]] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", order_by="TranscriptTurn.ordinal"
    )
    analysis: Mapped[MeetingAnalysis | None] = relationship(
        back_populates="meeting", cascade="all, delete-orphan", uselist=False
    )


class MeetingAnalysis(Base):
    """Grounded analysis result (summary/topics/decisions/actions/moments).

    One-to-one with a meeting. LLM failures only ever touch this table: the
    transcript, the meeting status and the audio artifacts stay untouched.
    """

    __tablename__ = "meeting_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), unique=True, index=True
    )
    status: Mapped[str] = mapped_column(String(16), default=ANALYSIS_STATUS_QUEUED, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_points_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    topics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    decisions_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    action_items_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    important_moments_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0)

    meeting: Mapped[Meeting] = relationship(back_populates="analysis")


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


class MeetingSpeakerAlias(Base):
    """Meeting-local display alias for one canonical speaker label.

    Display-only: no global person id, no embedding, no cross-meeting relation.
    Deleting the meeting removes its aliases (FK cascade).
    """

    __tablename__ = "meeting_speaker_aliases"
    __table_args__ = (
        UniqueConstraint("meeting_id", "canonical_speaker", name="uq_meeting_speaker_alias"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    canonical_speaker: Mapped[str] = mapped_column(String(32))
    display_name: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class MeetingLiveSpeaker(Base):
    """Canonical speaker timelines captured during a live ElevenLabs recording.

    Timestamps only (JSON list of [start, end] intervals) so the final full-file
    Scribe pass can map its own provider speakers onto the live canonical labels.
    Never audio, never embeddings.
    """

    __tablename__ = "meeting_live_speakers"
    __table_args__ = (
        UniqueConstraint("meeting_id", "canonical_speaker", name="uq_meeting_live_speaker"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(
        ForeignKey("meetings.id", ondelete="CASCADE"), index=True
    )
    canonical_speaker: Mapped[str] = mapped_column(String(32))
    intervals_json: Mapped[str] = mapped_column(Text)
