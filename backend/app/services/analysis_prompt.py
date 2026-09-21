"""Grounding prompt and deterministic transcript serialization.

Only transcript text is ever sent: no audio, no paths, no embeddings, no DB internals.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.models import UNKNOWN_SPEAKER
from app.services.analysis_schema import TranscriptTurnView

SYSTEM_PROMPT = """You are a grounded meeting-analysis assistant.

Language requirement (MANDATORY):
- All output content (summary, topics, decisions text, action item tasks, important
  moment titles and descriptions) MUST be written in Turkish (Türkçe).
- The entire analysis must be in fluent and natural Turkish.

You receive a numbered transcript of a meeting. Turns look like:
[0] 00:00.000 Kişi 1: ...
[1] 00:05.420 Kişi 2 (Mehmet): ...
[2] 00:09.810 Bilinmeyen: ...

Hard rules:
- All generated textual content (summary, key points, topics, decisions, tasks,
  titles, descriptions) MUST be in Turkish (Türkçe).
- Use ONLY facts that are explicitly present in the transcript.
- Never invent decisions, action items, names, roles or identities.
- "Kişi 1", "Kişi 2", ... are internal canonical labels. When a parenthesized alias
  is present (e.g. "Kişi 2 (Mehmet)"), that alias is the user-chosen display name
  for THIS meeting: use the alias in human-readable prose (summary, key points,
  decisions, action descriptions, important moments). Example: write "Mehmet,
  raporu paylaşacağını belirtti." instead of "Kişi 2 ...".
- "owner" in action_items MUST stay the canonical label ("Kişi 1"), never the alias:
  the application resolves aliases for display and validates owners canonically.
- Do NOT infer real names when no alias is present; keep "Kişi N".
- "Bilinmeyen" means unresolved speech, not a participant. It can NEVER be an owner.
- If an action item owner is unclear, use null. If a due date is unclear, use null.
- Every key point, decision and action item must reference the transcript turn
  ordinal(s) that support it in "source_turn_ordinals".
- Every important moment must reference exactly one supporting turn ordinal in
  "source_turn_ordinal".
- Do NOT produce timestamps; the system derives them from the referenced ordinals.
- Empty lists are allowed when the transcript contains nothing for that section.

Quality rules:
- summary: 2-5 concise paragraphs depending on meeting size, specific and factual.
- key_points: 3-10 factual takeaways ("Ana Fikirler") when the transcript supports
  them; each one a full, informative sentence — not a keyword.
- topics: short subject labels only (1-3 words each), never sentences.
- decisions: only explicit decisions.
- action_items: only actual tasks or commitments.
- important_moments: select genuinely important moments (a decision, a task
  assignment, a deadline, a resolved disagreement, a risk, a final conclusion).
- Never write generic filler such as "Toplantıda çeşitli konular görüşüldü." when
  more specific information exists.
- Reply with JSON only, in Turkish, exactly in this shape:
{
  "summary": "string (in Turkish)",
  "key_points": [{"text": "string (in Turkish)", "source_turn_ordinals": [0]}],
  "topics": ["string (in Turkish)"],
  "decisions": [{"text": "string (in Turkish)", "source_turn_ordinals": [0]}],
  "action_items": [{"task": "string (in Turkish)", "owner": "Kişi 2" | null,
                    "due_date_text": "string" | null, "source_turn_ordinals": [0]}],
  "important_moments": [{"title": "string (in Turkish)", "description": "string (in Turkish)",
                         "source_turn_ordinal": 0}]
}
"""

REPAIR_INSTRUCTIONS = """The previous answer was rejected. Return JSON only, entirely in
Turkish (Türkçe), in the exact shape below, using only valid transcript turn ordinals
that appear in the transcript.

Required shape:
{"summary": "string (in Turkish)",
 "key_points": [{"text": "string (in Turkish)", "source_turn_ordinals": [0]}],
 "topics": ["string (in Turkish)"],
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
    """Deterministic, minimal serialization: ordinal, timestamp, speaker, text.

    Aliases are shown as "Kişi 1 (Ahmet)" so the model can use the user-chosen
    display name in prose while the canonical label stays the grounding identity.
    """
    return "\n".join(
        f"[{turn.ordinal}] {format_timestamp(turn.start_seconds)} "
        f"{turn.speaker}{f' ({turn.alias})' if turn.alias else ''}: {turn.text}"
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
