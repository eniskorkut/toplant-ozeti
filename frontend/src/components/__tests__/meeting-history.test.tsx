import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { MeetingSummary } from "@/lib/api";
import { MeetingHistory } from "@/components/meeting-history";

const listMeetings = vi.fn();
const deleteMeeting = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listMeetings: (...args: unknown[]) => listMeetings(...args),
    deleteMeeting: (...args: unknown[]) => deleteMeeting(...args),
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
    transcription_provider: "elevenlabs",
    transcription_model: "scribe_v2",
  },
  {
    meeting_id: "older",
    created_at: "2026-09-17T09:00:00",
    duration_seconds: 42.5,
    status: "failed",
    requested_speaker_count: 2,
    has_transcript: false,
    analysis_status: null,
    transcription_provider: null,
    transcription_model: null,
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

  it("shows the provider badge and tolerates legacy null metadata", async () => {
    render(<MeetingHistory />);

    await screen.findAllByRole("listitem");
    expect(screen.getByText(/ElevenLabs/)).toBeTruthy();
    expect(screen.getAllByRole("listitem")[1].textContent).not.toContain("ElevenLabs");
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

async function openDeleteDialog(rowIndex = 0) {
  const buttons = await screen.findAllByRole("button", { name: /Toplantıyı sil/ });
  await act(async () => {
    fireEvent.click(buttons[rowIndex]);
  });
  return screen.getByRole("dialog");
}

describe("MeetingHistory deletion", () => {
  it("offers a delete action per row that only opens a confirmation", async () => {
    render(<MeetingHistory />);
    await screen.findAllByRole("listitem");

    const buttons = screen.getAllByRole("button", { name: /Toplantıyı sil/ });
    expect(buttons).toHaveLength(2);

    await act(async () => {
      fireEvent.click(buttons[0]);
    });

    expect(screen.getByRole("dialog")).toBeTruthy();
    expect(screen.getByText("Bu toplantı silinsin mi?")).toBeTruthy();
    expect(
      screen.getByText(/Ses kaydı, transkript ve analiz verileri kalıcı olarak silinecek/),
    ).toBeTruthy();
    expect(deleteMeeting).not.toHaveBeenCalled();
  });

  it("cancel closes the dialog and leaves the meeting untouched", async () => {
    render(<MeetingHistory />);
    await openDeleteDialog();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "İptal" }));
    });

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(deleteMeeting).not.toHaveBeenCalled();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("confirmed delete calls the API and removes the row immediately", async () => {
    deleteMeeting.mockResolvedValue(undefined);

    render(<MeetingHistory />);
    await openDeleteDialog(0);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Toplantıyı Sil" }));
    });

    expect(deleteMeeting).toHaveBeenCalledWith("newer");
    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1));
    expect(screen.getByText("Toplantı silindi.")).toBeTruthy();
  });

  it("keeps the row and shows the error when the API fails", async () => {
    deleteMeeting.mockRejectedValue(new ApiError(500, "sunucu hatası"));

    render(<MeetingHistory />);
    await openDeleteDialog(0);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Toplantıyı Sil" }));
    });

    expect(await screen.findByRole("alert")).toHaveTextContent("sunucu hatası");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("explains a 409 conflict for meetings that are still processing", async () => {
    deleteMeeting.mockRejectedValue(new ApiError(409, "Meeting is being processed"));

    render(<MeetingHistory />);
    await openDeleteDialog(1);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Toplantıyı Sil" }));
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(/işleniyor veya analiz ediliyor/);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("treats an already-deleted meeting (404) as removed", async () => {
    deleteMeeting.mockRejectedValue(new ApiError(404, "Meeting not found"));

    render(<MeetingHistory />);
    await openDeleteDialog(0);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Toplantıyı Sil" }));
    });

    await waitFor(() => expect(screen.getAllByRole("listitem")).toHaveLength(1));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
