import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { MeetingSummary } from "@/lib/api";
import { MeetingHistory } from "@/components/meeting-history";

const listMeetings = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listMeetings: (...args: unknown[]) => listMeetings(...args),
  };
});

const MEETINGS: MeetingSummary[] = [
  {
    meeting_id: "newer",
    created_at: "2026-09-18T12:00:00",
    duration_seconds: 95.1,
    status: "completed",
    requested_speaker_count: null,
    has_transcript: true,
    analysis_status: "completed",
  },
  {
    meeting_id: "older",
    created_at: "2026-09-17T09:00:00",
    duration_seconds: 42.5,
    status: "failed",
    requested_speaker_count: 2,
    has_transcript: false,
    analysis_status: null,
  },
];

afterEach(() => {
  vi.clearAllMocks();
});

beforeEach(() => {
  listMeetings.mockResolvedValue({ meetings: MEETINGS, count: MEETINGS.length });
});

describe("MeetingHistory", () => {
  it("renders meetings in the order the API returns (newest first)", async () => {
    render(<MeetingHistory />);

    const items = await screen.findAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0].textContent).toContain("Aç");
    const links = screen.getAllByRole("link", { name: "Aç" });
    expect(links[0].getAttribute("href")).toBe("/meetings/newer");
    expect(links[1].getAttribute("href")).toBe("/meetings/older");
  });

  it("shows status, analysis status and known speaker count", async () => {
    render(<MeetingHistory />);

    await screen.findAllByRole("listitem");
    expect(screen.getByText(/Tamamlandı · Analiz: Tamamlandı/)).toBeTruthy();
    expect(screen.getByText(/Başarısız/)).toBeTruthy();
    expect(screen.getByText(/2 konuşmacı/)).toBeTruthy();
  });

  it("shows a clear empty state", async () => {
    listMeetings.mockResolvedValue({ meetings: [], count: 0 });

    render(<MeetingHistory />);

    expect(await screen.findByText(/Henüz toplantı yok/)).toBeTruthy();
  });

  it("shows an error without throwing", async () => {
    listMeetings.mockRejectedValue(new ApiError(500, "sunucu hatası"));

    render(<MeetingHistory />);

    expect(await screen.findByRole("alert")).toHaveTextContent("sunucu hatası");
  });

  it("reloads when Yenile is pressed", async () => {
    render(<MeetingHistory />);
    await screen.findAllByRole("listitem");

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Yenile" }));
    });

    expect(listMeetings).toHaveBeenCalledTimes(2);
  });
});
