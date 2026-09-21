/** Deterministic clipboard notes built from the structured analysis response. */

import type { MeetingAnalysis } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";
import { resolveSpeakerDisplay, type SpeakerAliases } from "@/lib/speakers";

function timestamp(seconds: number): string {
  return formatTimestamp(seconds);
}

export function buildMeetingNotes(
  analysis: MeetingAnalysis,
  aliases: SpeakerAliases,
): string {
  const lines: string[] = [];

  lines.push("TOPLANTI ÖZETİ", "", (analysis.summary ?? "").trim(), "");

  if (analysis.key_points.length > 0) {
    lines.push("ANA FİKİRLER", "");
    for (const point of analysis.key_points) {
      lines.push(`- [${timestamp(point.timestamp_seconds)}] ${point.text}`);
    }
    lines.push("");
  }

  if (analysis.decisions.length > 0) {
    lines.push("KARARLAR", "");
    for (const decision of analysis.decisions) {
      lines.push(`- [${timestamp(decision.timestamp_seconds)}] ${decision.text}`);
    }
    lines.push("");
  }

  if (analysis.action_items.length > 0) {
    lines.push("ALINACAK AKSİYONLAR", "");
    for (const item of analysis.action_items) {
      const owner = item.owner ? resolveSpeakerDisplay(item.owner, aliases) : null;
      const head = `- [${timestamp(item.timestamp_seconds)}] ${owner ? `${owner} — ` : ""}${item.task}`;
      lines.push(item.due_date_text ? `${head} · ${item.due_date_text}` : head);
    }
    lines.push("");
  }

  if (analysis.important_moments.length > 0) {
    lines.push("ÖNEMLİ ANLAR", "");
    for (const moment of analysis.important_moments) {
      lines.push(`- [${timestamp(moment.timestamp_seconds)}] ${moment.title}`);
      if (moment.description) {
        lines.push(`  ${moment.description}`);
      }
    }
    lines.push("");
  }

  return `${lines.join("\n").trimEnd()}\n`;
}
