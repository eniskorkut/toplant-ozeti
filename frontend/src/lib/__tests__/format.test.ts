import { describe, expect, it } from "vitest";

import { DISPLAY_TIME_ZONE, formatDateTime, formatMeetingDate, formatMeetingDateShort } from "@/lib/format";

describe("meeting time formatting", () => {
  it("is pinned to Europe/Istanbul regardless of the runner timezone", () => {
    expect(DISPLAY_TIME_ZONE).toBe("Europe/Istanbul");
    // 11:05 UTC is 14:05 in Istanbul (UTC+3).
    const instant = "2026-09-19T11:05:40.494864+00:00";
    expect(formatMeetingDate(instant)).toContain("14:05");
    expect(formatMeetingDateShort(instant)).toContain("14:05");
    expect(formatDateTime(instant)).toContain("14:05");
  });

  it("treats naive UTC strings from old rows as UTC (the API now adds +00:00)", () => {
    expect(formatMeetingDate("2026-09-19T11:05:40")).toContain("14:05");
  });

  it("renders the Turkish date style", () => {
    expect(formatMeetingDate("2026-09-18T17:03:00+00:00")).toMatch(/18 Eylül 2026/);
  });
});
