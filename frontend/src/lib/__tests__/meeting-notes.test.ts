import { describe, expect, it } from "vitest";

import type { MeetingAnalysis } from "@/lib/api";
import { buildMeetingNotes } from "@/lib/meeting-notes";

const ANALYSIS: MeetingAnalysis = {
  meeting_id: "m1",
  status: "completed",
  provider: "mock",
  model: "mock-model",
  summary: "Ekip yayın planını netleştirdi.",
  key_points: [
    { text: "Yayın cuma günü yapılacak.", source_turn_ordinals: [1], timestamp_seconds: 12.4 },
    { text: "Testler perşembe bitecek.", source_turn_ordinals: [2], timestamp_seconds: 3725 },
  ],
  topics: ["yayın"],
  decisions: [
    { text: "Cuma yayınlanacak.", source_turn_ordinals: [1], timestamp_seconds: 12.4 },
  ],
  action_items: [
    {
      task: "Raporu paylaş",
      owner: "Kişi 1",
      due_date_text: "cuma",
      source_turn_ordinals: [1],
      timestamp_seconds: 322,
    },
    {
      task: "Testleri tamamla",
      owner: null,
      due_date_text: null,
      source_turn_ordinals: [2],
      timestamp_seconds: 490,
    },
  ],
  important_moments: [
    {
      title: "Yayın kararı",
      description: "Testler başarılı olursa cuma yayın.",
      source_turn_ordinal: 1,
      timestamp_seconds: 134,
    },
  ],
  analysis_error: null,
  input_chars: 100,
  latency_seconds: 1.0,
  repair_attempts: 0,
};

describe("buildMeetingNotes", () => {
  it("emits all headings in the required order without JSON", () => {
    const text = buildMeetingNotes(ANALYSIS, {});
    const order = [
      "TOPLANTI ÖZETİ",
      "ANA FİKİRLER",
      "KARARLAR",
      "ALINACAK AKSİYONLAR",
      "ÖNEMLİ ANLAR",
    ];
    const positions = order.map((heading) => text.indexOf(heading));
    expect(positions.every((position) => position >= 0)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
    expect(text).not.toContain("{");
  });

  it("resolves aliases for owners and keeps canonical names otherwise", () => {
    const text = buildMeetingNotes(ANALYSIS, { "Kişi 1": "Ahmet" });
    expect(text).toContain("Ahmet — Raporu paylaş · cuma");
    expect(text).toContain("- [08:10] Testleri tamamla");
  });

  it("formats timestamps as mm:ss and hh:mm:ss", () => {
    const text = buildMeetingNotes(ANALYSIS, {});
    expect(text).toContain("[00:12]");
    expect(text).toContain("[1:02:05]"); // 3725 seconds
    expect(text).toContain("[02:14]");
  });

  it("handles empty sections cleanly", () => {
    const text = buildMeetingNotes(
      { ...ANALYSIS, key_points: [], decisions: [], action_items: [], important_moments: [] },
      {},
    );
    expect(text).toContain("TOPLANTI ÖZETİ");
    expect(text).not.toContain("KARARLAR");
  });
});
