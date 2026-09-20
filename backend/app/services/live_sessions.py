"""Ephemeral live-session state for delayed long-context speaker attribution.

Design (measured): short independent windows under-attribute rapid conversational
diarization, while the full-file Scribe pass separates speakers well. The live path
therefore sends LONG overlapping snapshots (lookback + small step) and turns them
into stable canonical labels through:

- previous-snapshot matching (request-local ids are never compared by name);
- promotion evidence: an unmatched cluster must persist across snapshots (or be
  part of a sufficiently long first snapshot) before it becomes a new Kişi N;
- stable region / mutable tail: only the part of a snapshot older than one step is
  confirmed, the newest seconds stay provisional;
- dense numbering and a per-session registry that never survives the recording.

In-memory only: timestamps, labels and aliases. No audio, no embeddings, no
provider payloads.
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
MAX_INTERVALS_PER_SPEAKER = 4_000
# Every canonical speaker needs evidence from at least this many overlapping
# snapshots. One snapshot alone is never enough: that is what used to create
# runaway Kişi 5/8/10 labels.
MIN_CANDIDATE_SNAPSHOTS = 2
# Conservative confirmation: an existing canonical speaker is only confirmed
# when matched via temporal overlap with a decisive margin against competing
# candidates. Weak disjoint merges or pause gaps remain provisional until
# confirmed by subsequent overlapping snapshots or full-file finalization.
CONFIRM_MARGIN_SECONDS = 1.0
CONFIRM_MIN_AGREEING_SNAPSHOTS = 2
MAX_HISTORY_ENTRIES = 600


@dataclass
class LiveSpeaker:
    label: str
    # Confirmed (stable) intervals only; never relabeled again.
    committed: list[Interval] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return total_seconds(self.committed)


@dataclass
class CandidateSpeaker:
    """Unmatched provider cluster waiting for promotion evidence."""

    intervals: list[Interval] = field(default_factory=list)
    seen_snapshots: int = 0
    first_sequence: int = 0
    last_sequence: int = 0

    @property
    def speech_seconds(self) -> float:
        return total_seconds(self.intervals)


@dataclass
class LiveSession:
    id: str
    created_at: float
    last_seen: float
    next_sequence: int = 1
    windows_received: int = 0
    rolling_seconds: float = 0.0
    label_switches: int = 0
    ambiguous_segments: int = 0
    # Known speaker count from the user selection (num_speakers = maximum expected).
    max_speakers: int | None = None
    speakers: dict[str, LiveSpeaker] = field(default_factory=dict)
    # Labels with at least one conservatively confirmed span (alias-eligible).
    confirmed_labels_set: set[str] = field(default_factory=set)
    candidates: list[CandidateSpeaker] = field(default_factory=list)
    # Full spans of the previous snapshot per canonical label (matching reference).
    last_snapshot: dict[str, list[Interval]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    # Calibration history: every stable matched span with its confidence inputs
    # (timestamps and scores only, never text). Used by the offline policy study.
    assignment_history: list[dict] = field(default_factory=list)

    def speaker(self, label: str) -> LiveSpeaker:
        speaker = self.speakers.get(label)
        if speaker is None:
            speaker = LiveSpeaker(label=label)
            self.speakers[label] = speaker
        return speaker

    def confirmed_labels(self) -> list[str]:
        return sorted(self.confirmed_labels_set, key=_label_sort_key)

    def public_state(self) -> dict:
        return {
            "live_session_id": self.id,
            "windows_received": self.windows_received,
            "rolling_seconds": round(self.rolling_seconds, 3),
            "label_switches": self.label_switches,
            "ambiguous_segments": self.ambiguous_segments,
            "candidate_speakers": len(self.candidates),
            "confirmed_speakers": self.confirmed_labels(),
            "speakers": [
                {
                    "canonical_speaker": label,
                    "display_name": self.aliases.get(label, label),
                    "speech_seconds": round(speaker.total_seconds, 3),
                }
                for label, speaker in sorted(
                    self.speakers.items(), key=lambda item: _label_sort_key(item[0])
                )
            ],
            "aliases": dict(self.aliases),
        }


def _label_sort_key(label: str) -> tuple[int, str]:
    if label.startswith("Kişi "):
        try:
            return (int(label.split(" ", 1)[1]), label)
        except ValueError:
            return (999, label)
    return (1000, label)


class LiveSessionStore:
    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._sessions: dict[str, LiveSession] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def create(self, max_speakers: int | None = None) -> LiveSession:
        now = time.monotonic()
        session = LiveSession(
            id=uuid.uuid4().hex, created_at=now, last_seen=now, max_speakers=max_speakers
        )
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

    # --- rolling snapshots -------------------------------------------------

    def apply_window(
        self,
        session: LiveSession,
        *,
        provider_intervals: dict[str, list[Interval]],
        window: Interval,
        sequence: int,
        stable_until: float | None = None,
    ) -> dict:
        """Fold one long-context snapshot into the session.

        `stable_until` marks the boundary between the confirmable region and the
        mutable tail (newest step). Matched spans are split accordingly; only the
        stable part is committed, the tail is reported as provisional.
        """
        boundary = window[1] if stable_until is None else min(
            max(stable_until, window[0]), window[1]
        )

        reference = session.last_snapshot or {
            label: list(speaker.committed) for label, speaker in session.speakers.items()
        }
        matches, unmatched, ambiguous = match_speakers(
            reference,
            provider_intervals,
            window=window,
            gap_tolerance_seconds=DEFAULT_GAP_TOLERANCE_SECONDS,
            allow_disjoint_merge=True,
        )

        assignments: list[dict] = []
        new_speakers: list[str] = []
        last_snapshot: dict[str, list[Interval]] = {}

        def split(spans: list[Interval]) -> tuple[list[Interval], list[Interval]]:
            stable = [
                (max(start, window[0]), min(end, boundary))
                for start, end in spans
                if start < boundary
            ]
            tail = [
                (max(start, boundary), min(end, window[1]))
                for start, end in spans
                if end > boundary
            ]
            return [span for span in stable if span[1] > span[0]], [
                span for span in tail if span[1] > span[0]
            ]

        def record(label: str, spans: list[Interval]) -> None:
            last_snapshot.setdefault(label, []).extend(spans)

        consumed_candidates: list[int] = []
        for provider_name in sorted(provider_intervals):
            spans = provider_intervals[provider_name]
            if provider_name in ambiguous:
                session.ambiguous_segments += 1
                continue
            match: SpeakerMatch | None = matches.get(provider_name)
            if match is None:
                self._register_candidate(session, spans, sequence)
                continue
            label = match.canonical_speaker
            stable, tail = split(spans)
            speaker = session.speaker(label)
            speaker.committed.extend(stable)
            speaker.committed = speaker.committed[-MAX_INTERVALS_PER_SPEAKER:]
            record(label, spans)
            for span in stable:
                session.assignment_history.append(
                    {
                        "sequence": sequence,
                        "label": label,
                        "start": round(span[0], 3),
                        "end": round(span[1], 3),
                        "confidence": match.ratio,
                        "margin": match.margin,
                        "evidence": match.evidence,
                        "provider_speakers": len(provider_intervals),
                        "coexist": len(provider_intervals) > 1,
                        "window_end": round(window[1], 3),
                    }
                )
            session.assignment_history = session.assignment_history[-MAX_HISTORY_ENTRIES:]
            if stable:
                confirmed = self._is_confirmed(session, label, stable, match=match)
                if confirmed:
                    session.confirmed_labels_set.add(label)
                assignments.append(
                    {
                        "canonical_speaker": label,
                        "is_new": False,
                        "confidence": match.ratio,
                        "evidence": match.evidence,
                        "provisional": not confirmed,
                        "start": round(min(start for start, _ in stable), 3),
                        "end": round(max(end for _, end in stable), 3),
                        "speech_seconds": round(total_seconds(stable), 3),
                    }
                )
            if tail:
                assignments.append(
                    {
                        "canonical_speaker": label,
                        "is_new": False,
                        "confidence": match.ratio,
                        "evidence": match.evidence,
                        "provisional": True,
                        "start": round(min(start for start, _ in tail), 3),
                        "end": round(max(end for _, end in tail), 3),
                        "speech_seconds": round(total_seconds(tail), 3),
                    }
                )

        # Promotion: only clusters confirmed by multiple overlapping snapshots
        # become canonical speakers (first appearance wins, numbering stays dense).
        promoted: list[str] = []
        for index, candidate in enumerate(session.candidates):
            if not self._may_promote(session):
                break
            if candidate.seen_snapshots < MIN_CANDIDATE_SNAPSHOTS:
                continue
            label = next_canonical_label(set(session.speakers))
            speaker = session.speaker(label)
            stable, tail = split(candidate.intervals)
            speaker.committed.extend(stable)
            record(label, candidate.intervals)
            promoted.append(label)
            new_speakers.append(label)
            consumed_candidates.append(index)
            session.assignment_history.extend(
                {
                    "sequence": sequence,
                    "label": label,
                    "start": round(span[0], 3),
                    "end": round(span[1], 3),
                    "confidence": None,
                    "margin": 0.0,
                    "evidence": "promoted",
                    "provider_speakers": len(provider_intervals),
                    "coexist": len(provider_intervals) > 1,
                    "window_end": round(window[1], 3),
                }
                for span in stable
            )
            session.assignment_history = session.assignment_history[-MAX_HISTORY_ENTRIES:]
            if stable:
                session.confirmed_labels_set.add(label)
                assignments.append(
                    {
                        "canonical_speaker": label,
                        "is_new": True,
                        "confidence": None,
                        "evidence": "promoted",
                        "provisional": False,
                        "start": round(min(start for start, _ in stable), 3),
                        "end": round(max(end for _, end in stable), 3),
                        "speech_seconds": round(total_seconds(stable), 3),
                    }
                )
            if tail:
                assignments.append(
                    {
                        "canonical_speaker": label,
                        "is_new": True,
                        "confidence": None,
                        "evidence": "promoted",
                        "provisional": True,
                        "start": round(min(start for start, _ in tail), 3),
                        "end": round(max(end for _, end in tail), 3),
                        "speech_seconds": round(total_seconds(tail), 3),
                    }
                )
        session.candidates = [
            candidate
            for index, candidate in enumerate(session.candidates)
            if index not in consumed_candidates
        ]
        session.last_snapshot = last_snapshot
        session.windows_received += 1
        session.rolling_seconds += max(0.0, window[1] - window[0])
        if session.windows_received > 1 and new_speakers:
            session.label_switches += len(new_speakers)

        return {
            "sequence": sequence,
            "window": [round(window[0], 3), round(window[1], 3)],
            "assignments": assignments,
            "new_speakers": new_speakers,
            "promoted_speakers": promoted,
            "ambiguous_speakers": len(ambiguous),
            "candidate_speakers": len(session.candidates),
            "confirmed_speakers": session.confirmed_labels(),
        }

    def _is_confirmed(
        self,
        session: LiveSession,
        label: str,
        stable: list[Interval],
        *,
        match: SpeakerMatch | None = None,
    ) -> bool:
        """Conservative confirmation gate.

        A live attribution is only confirmed when:
        1. It is a newly promoted candidate with multi-snapshot evidence (seen
           across >= 2 snapshots).
        2. Or it is an overlap match with a clear margin (>= CONFIRM_MARGIN_SECONDS)
           against competing canonical candidates, and not a weak disjoint merge or pause gap.

        Everything else stays provisional ("Konuşmacı belirleniyor") until confirmed or finalized
        by the authoritative full-file Scribe pass.
        """
        if match is None:
            return True
        if match.evidence in ("merged-fragment", "gap"):
            return False
        return match.margin >= CONFIRM_MARGIN_SECONDS

    def _may_promote(self, session: LiveSession) -> bool:
        if session.max_speakers is None:
            return True
        return len(session.speakers) < session.max_speakers

    def _register_candidate(
        self, session: LiveSession, spans: list[Interval], sequence: int
    ) -> None:
        """Accumulate promotion evidence for an unmatched cluster."""
        if not spans:
            return
        for candidate in session.candidates:
            # Evidence must come from an EARLIER snapshot: two clusters inside the
            # same snapshot are different voices, not one candidate.
            if candidate.last_sequence == sequence:
                continue
            if _clusters_are_continuous(candidate.intervals, spans):
                candidate.intervals.extend(spans)
                candidate.intervals = candidate.intervals[-MAX_INTERVALS_PER_SPEAKER:]
                candidate.seen_snapshots += 1
                candidate.last_sequence = sequence
                if candidate.first_sequence == 0:
                    candidate.first_sequence = sequence
                return
        session.candidates.append(
            CandidateSpeaker(
                intervals=list(spans),
                seen_snapshots=1,
                first_sequence=sequence,
                last_sequence=sequence,
            )
        )


def _clusters_are_continuous(
    candidate_intervals: list[Interval], spans: list[Interval], tolerance: float = 2.0
) -> bool:
    """Same cluster when the spans overlap or sit within a short conversational gap."""
    for left in candidate_intervals:
        for right in spans:
            if left[1] < right[0]:
                gap = right[0] - left[1]
            elif right[1] < left[0]:
                gap = left[0] - right[1]
            else:
                return True
            if gap <= tolerance:
                return True
    return False


MAX_INTERVALS_PER_SNAPSHOT = MAX_INTERVALS_PER_SPEAKER


@lru_cache
def get_live_session_store() -> LiveSessionStore:
    return LiveSessionStore()
