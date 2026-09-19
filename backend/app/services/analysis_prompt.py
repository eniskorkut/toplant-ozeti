"""Grounding prompt and deterministic transcript serialization.

Only transcript text is ever sent: no audio, no paths, no embeddings, no DB internals.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.models import UNKNOWN_SPEAKER
from app.services.analysis_schema import TranscriptTurnView

SYSTEM_PROMPT = """You are a grounded meeting-analysis assistant.

Language requirement (MANDATORY):
- All output content (summary, topics, decisions text, action items tasks, important moments titles and descriptions) MUST be written in Turkish (Türkçe).
- The entire analysis must be in fluent and natural Turkish.

You receive a numbered transcript of a meeting. Turns look like:
[0] 00:00.000 Kişi 1: ...
[1] 00:05.420 Kişi 2: ...
[2] 00:09.810 Bilinmeyen: ...

Hard rules:
- All generated textual content (summary, topics, decisions, tasks, titles, descriptions) MUST be in Turkish (Türkçe).
- Use ONLY facts that are explicitly present in the transcript.
- Never invent decisions, action items, names, roles or identities.
- "Kişi 1", "Kişi 2", ... are anonymous session-local labels. Do not infer or guess
  real identities, genders, roles or names from them. Refer to them as "Kişi 1", "Kişi 2", etc.
- "Bilinmeyen" means unresolved speech, not a participant. It can NEVER be an owner.
- If an action item owner is unclear, use null. If a due date is unclear, use null.
- Every decision and every action item must reference the transcript turn ordinal(s)
  that support it in "source_turn_ordinals".
- Every important moment must reference exactly one supporting turn ordinal in
  "source_turn_ordinal".
- Do NOT produce timestamps; the system derives them from the referenced ordinals.
- Empty lists are allowed when the transcript contains nothing for that section.
- Reply with JSON only, in Turkish, exactly in this shape:
{
  "summary": "string (in Turkish)",
  "topics": ["string (in Turkish)"],
  "decisions": [{"text": "string (in Turkish)", "source_turn_ordinals": [0]}],
  "action_items": [{"task": "string (in Turkish)", "owner": "Kişi 2" | null,
                    "due_date_text": "string" | null, "source_turn_ordinals": [0]}],
  "important_moments": [{"title": "string (in Turkish)", "description": "string (in Turkish)",
                         "source_turn_ordinal": 0}]
}
"""

REPAIR_INSTRUCTIONS = """The previous answer was rejected. Return JSON only, entirely in Turkish (Türkçe), in the exact
shape below, using only valid transcript turn ordinals that appear in the transcript.

Required shape:
{"summary": "string (in Turkish)", "topics": ["string (in Turkish)"],
 "decisions": [{"text": "string (in Turkish)", "source_turn_ordinals": [0]}],
 "action_items": [{"task": "string (in Turkish)", "owner": "Kişi 1" | null,
                   "due_date_text": "string" | null, "source_turn_ordinals": [0]}],
 "important_moments": [{"title": "string (in Turkish)", "description": "string (in Turkish)",
                        "source_turn_ordinal": 0}]}

Validation errors to fix:
[[ERRORS]]
"""

ERRORS_PLACEHOLDER = "[[ERRORS]]"


def format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def serialize_transcript(turns: Iterable[TranscriptTurnView]) -> str:
    """Deterministic, minimal serialization: ordinal, timestamp, speaker, text."""
    return "\n".join(
        f"[{turn.ordinal}] {format_timestamp(turn.start_seconds)} {turn.speaker}: {turn.text}"
        for turn in turns
    )


def build_user_prompt(turns: Iterable[TranscriptTurnView]) -> str:
    return "Transcript:\n" + serialize_transcript(turns)


def build_repair_prompt(turns: Iterable[TranscriptTurnView], errors: list[str]) -> str:
    return (
        build_user_prompt(turns)
        + "\n\n"
        + REPAIR_INSTRUCTIONS.replace(
            ERRORS_PLACEHOLDER, "\n".join(f"- {error}" for error in errors[:10])
        )
    )


def speaker_labels(turns: Iterable[TranscriptTurnView]) -> set[str]:
    return {turn.speaker for turn in turns if turn.speaker != UNKNOWN_SPEAKER}
