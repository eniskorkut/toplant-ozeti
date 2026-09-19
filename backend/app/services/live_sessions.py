"""Ephemeral live-session state for the ElevenLabs live meeting experience.

In-memory only: a live session is disposable state for one ongoing recording. It
holds no audio, no embeddings and no provider payloads — only canonical speaker
timelines (timestamps), display aliases and safe counters. Sessions expire on a
TTL so an abandoned recording cannot leak state forever.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache

from app.services.speaker_matching import (
    DEFAULT_GAP_TOLERANCE_SECONDS,
    Interval,
    SpeakerMatch,
    match_speakers,
    next_canonical_label,
    total_seconds,
)

DEFAULT_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 32
MAX_INTERVALS_PER_SPEAKER = 2_000


@dataclass
class LiveSpeaker:
    label: str
    # Windows that are older than the newest one; never relabeled again.
    committed: list[Interval] = field(default_factory=list)
    # Intervals of the newest window only: the small mutable overlap tail.
    latest: list[Interval] = field(default_factory=list)

    def intervals(self) -> list[Interval]:
        return self.committed + self.latest

    @property
    def latest_seconds(self) -> float:
        return total_seconds(self.latest)

    @property
    def total_seconds(self) -> float:
        return total_seconds(self.committed) + self.latest_seconds


@dataclass
class LiveSession:
    id: str
    created_at: float
    last_seen: float
    next_sequence: int = 1
    windows_received: int = 0
    rolling_seconds: float = 0.0
    label_switches: int = 0
    revision_events: int = 0
    ambiguous_segments: int = 0
    pending_intervals: list[Interval] = field(default_factory=list)
    speakers: dict[str, LiveSpeaker] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)

    def speaker(self, label: str) -> LiveSpeaker:
        speaker = self.speakers.get(label)
        if speaker is None:
            speaker = LiveSpeaker(label=label)
            self.speakers[label] = speaker
        return speaker

    def canonical_intervals(self) -> dict[str, list[Interval]]:
        return {label: speaker.intervals() for label, speaker in self.speakers.items()}

    def public_state(self) -> dict:
        return {
            "live_session_id": self.id,
            "windows_received": self.windows_received,
            "rolling_seconds": round(self.rolling_seconds, 3),
            "label_switches": self.label_switches,
            "revision_events": self.revision_events,
            "ambiguous_segments": self.ambiguous_segments,
            "pending_seconds": round(total_seconds(self.pending_intervals), 3),
            "speakers": [
                {
                    "canonical_speaker": label,
                    "display_name": self.aliases.get(label, label),
                    "speech_seconds": round(speaker.total_seconds, 3),
                }
                for label, speaker in sorted(self.speakers.items())
            ],
            "aliases": dict(self.aliases),
        }


class LiveSessionStore:
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    # --- lifecycle ---------------------------------------------------------

    def create(self) -> LiveSession:
        now = time.monotonic()
        session = LiveSession(id=uuid.uuid4().hex, created_at=now, last_seen=now)
        with self._lock:
            self._prune_locked(now)
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda item: item.last_seen)
                self._sessions.pop(oldest.id, None)
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> LiveSession | None:
        now = time.monotonic()
        with self._lock:
            self._prune_locked(now)
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = now
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        with self._lock:
            self._prune_locked(time.monotonic())
            return len(self._sessions)

    def _prune_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_seen > self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    # --- rolling windows ---------------------------------------------------

    def apply_window(
        self,
        session: LiveSession,
        *,
        provider_intervals: dict[str, list[Interval]],
        window: Interval,
        sequence: int,
        min_overlap_seconds: float | None = None,
        min_overlap_ratio: float | None = None,
    ) -> dict:
        """Match one request-local result onto canonical speakers and store it.

        The newest window's intervals form a mutable tail: they may be revised by
        the next window, while older windows stay committed. This is what keeps a
        Kişi label from flipping for the whole history.
        """
        kwargs: dict[str, float] = {}
        if min_overlap_seconds is not None:
            kwargs["min_overlap_seconds"] = min_overlap_seconds
        if min_overlap_ratio is not None:
            kwargs["min_overlap_ratio"] = min_overlap_ratio

        # Match against the PREVIOUS window's own intervals, not the accumulated
        # history: overlapping windows share audio, so word timings in the shared
        # region are direct evidence and a single wrong assignment cannot poison
        # every later window. When a speaker did not appear in the previous
        # window, fall back to their committed history (resumption after a gap).
        reference = {
            label: (list(speaker.latest) if speaker.latest else list(speaker.committed))
            for label, speaker in session.speakers.items()
        }
        matches, unmatched, ambiguous = match_speakers(
            reference,
            provider_intervals,
            window=window,
            gap_tolerance_seconds=DEFAULT_GAP_TOLERANCE_SECONDS,
            allow_disjoint_merge=True,
            **kwargs,
        )

        # Previous tail becomes committed history before the new window is applied.
        for speaker in session.speakers.values():
            speaker.committed.extend(speaker.latest)
            speaker.latest = []

        existing_labels = set(session.speakers)
        new_speakers: list[str] = []

        assignments: list[dict] = []
        for provider_name in sorted(provider_intervals):
            spans = provider_intervals[provider_name]
            if provider_name in ambiguous:
                # Evidence exists but is not decisive: keep it provisional and do
                # not pollute any canonical timeline with a guess.
                session.pending_intervals.extend(spans)
                continue
            match: SpeakerMatch | None = matches.get(provider_name)
            if match is None:
                label = next_canonical_label(existing_labels)
                existing_labels.add(label)
                new_speakers.append(label)
                confidence = None
                evidence = "new"
            else:
                label = match.canonical_speaker
                confidence = match.ratio
                evidence = match.evidence
            speaker = session.speaker(label)
            speaker.latest.extend(spans)
            assignments.append(
                {
                    "canonical_speaker": label,
                    "is_new": label in new_speakers,
                    "confidence": confidence,
                    "evidence": evidence,
                    "start": round(min(start for start, _ in spans), 3),
                    "end": round(max(end for _, end in spans), 3),
                    "speech_seconds": round(total_seconds(spans), 3),
                }
            )
        session.ambiguous_segments += len(ambiguous)

        session.windows_received += 1
        session.rolling_seconds += max(0.0, window[1] - window[0])
        if session.windows_received > 1 and new_speakers:
            # New canonical speakers after the first window are fragmentation
            # evidence (stable meetings should not allocate more labels).
            session.label_switches += len(new_speakers)
        for speaker in session.speakers.values():
            if len(speaker.committed) > MAX_INTERVALS_PER_SPEAKER:
                speaker.committed = speaker.committed[-MAX_INTERVALS_PER_SPEAKER:]

        return {
            "sequence": sequence,
            "window": [round(window[0], 3), round(window[1], 3)],
            "assignments": assignments,
            "new_speakers": new_speakers,
            "ambiguous_speakers": len(ambiguous),
        }


@lru_cache
def get_live_session_store() -> LiveSessionStore:
    return LiveSessionStore()
