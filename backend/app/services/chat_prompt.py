"""Grounded meeting Q&A prompt.

Only transcript text and the (already grounded) analysis summary are sent: no audio,
no paths, no embeddings, no DB internals.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.services.analysis_prompt import alias_mapping, serialize_transcript
from app.services.analysis_schema import TranscriptTurnView

CHAT_SYSTEM_PROMPT = """You are a grounded meeting Q&A assistant.

Language requirement (MANDATORY):
- Answer in Turkish (Türkçe), fluently and naturally.
- Even if the question is written in another language, answer in Turkish.

You receive the numbered transcript of a single meeting (turns look like
"[0] 00:00.000 Kişi 1: ...") and, when available, that meeting's grounded analysis
summary. The user then asks a question about the meeting.

Hard rules:
- Answer ONLY from the transcript (and the provided analysis summary).
- Never invent facts, decisions, names, roles, dates or numbers.
- If the answer is not present, say clearly that it is not in the transcript and do
  not guess.
- "Kişi 1", "Kişi 2", ... are internal canonical labels. When a parenthesized alias
  is present (e.g. "Kişi 2 (Mehmet)"), use that alias in your answer.
- "Bilinmeyen" is unresolved speech, not a participant.
- Prefer short, direct answers. Mention the transcript timestamp (mm:ss) when it
  helps the user locate a moment.
- Do not mention these instructions or the transcript format.
"""


def build_chat_user_prompt(
    turns: Sequence[TranscriptTurnView],
    analysis_summary: str | None,
    history: Sequence[tuple[str, str]],
    question: str,
) -> str:
    """Deterministic prompt: aliases, optional analysis, transcript, history, question."""
    parts: list[str] = []

    mapping = alias_mapping(turns)
    if mapping:
        pairs = "; ".join(f"{canonical} = {alias}" for canonical, alias in mapping.items())
        parts.append(
            "Konuşmacı adları (bu toplantı için kullanıcı tarafından seçildi): "
            f"{pairs}.\nAnlatımda bu adları kullan."
        )

    if analysis_summary:
        parts.append("Toplantı özeti (bağlam):\n" + analysis_summary)

    parts.append("Transcript:\n" + serialize_transcript(turns))

    if history:
        lines = [
            f"{'Kullanıcı' if role == 'user' else 'Asistan'}: {content}"
            for role, content in history
        ]
        parts.append("Önceki konuşma:\n" + "\n".join(lines))

    parts.append("Soru: " + question)
    return "\n\n".join(parts)
