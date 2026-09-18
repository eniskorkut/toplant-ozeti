import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { MeetingAnalysis, MeetingStatus, Transcript } from "@/lib/api";
import { MeetingDetail } from "@/components/meeting-detail";

const getMeeting = vi.fn();
const getTranscript = vi.fn();
const getAnalysis = vi.fn();
const startAnalysis = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getMeeting: (...args: unknown[]) => getMeeting(...args),
    getTranscript: (...args: unknown[]) => getTranscript(...args),
    getAnalysis: (...args: unknown[]) => getAnalysis(...args),
    startAnalysis: (...args: unknown[]) => startAnalysis(...args),
    meetingAudioUrl: (id: string) => `http://localhost:8000/api/v1/meetings/${id}/audio`,
  };
});

const COMPLETED_MEETING: MeetingStatus = {
  meeting_id: "m1",
  status: "completed",
  created_at: "2026-09-18T10:00:00",
  duration_seconds: 42.5,
  requested_speaker_count: null,
  processing_error: null,
  has_transcript: true,
};

const TRANSCRIPT: Transcript = {
  meeting_id: "m1",
  status: "completed",
  duration_seconds: 42.5,
  speakers: ["Kişi 1", "Kişi 2"],
  unresolved_label: "Bilinmeyen",
  unresolved_turns: 1,
  turns: [
    { ordinal: 0, speaker: "Kişi 1", start_seconds: 0, end_seconds: 4, text: "Merhaba" },
    { ordinal: 1, speaker: "Kişi 2", start_seconds: 12.35, end_seconds: 15, text: "Tamam" },
    { ordinal: 2, speaker: "Bilinmeyen", start_seconds: 20, end_seconds: 21, text: "???" },
  ],
};

const COMPLETED_ANALYSIS: MeetingAnalysis = {
  meeting_id: "m1",
  status: "completed",
  provider: "mock",
  model: "mock-model",
  summary: "Toplantı özeti.",
  topics: ["konu bir", "konu iki"],
  decisions: [{ text: "Karar verildi.", source_turn_ordinals: [1], timestamp_seconds: 12.35 }],
  action_items: [
    {
      task: "Rapor yazılacak.",
      owner: "Kişi 2",
      due_date_text: "cuma",
      source_turn_ordinals: [1],
      timestamp_seconds: 12.35,
    },
  ],
  important_moments: [
    {
      title: "Önemli an",
      description: "Açıklama",
      source_turn_ordinal: 2,
      timestamp_seconds: 20,
    },
  ],
  analysis_error: null,
  input_chars: 100,
  latency_seconds: 1.2,
  repair_attempts: 0,
};

beforeEach(() => {
  getMeeting.mockResolvedValue(COMPLETED_MEETING);
  getTranscript.mockResolvedValue(TRANSCRIPT);
  getAnalysis.mockRejectedValue(new ApiError(404, "Analysis not found"));
  startAnalysis.mockResolvedValue({ ...COMPLETED_ANALYSIS, status: "queued" });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("MeetingDetail", () => {
  it("loads everything from the backend on refresh (no local memory needed)", async () => {
    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("Merhaba")).toBeTruthy();
    expect(getMeeting).toHaveBeenCalledWith("m1");
    expect(getTranscript).toHaveBeenCalledWith("m1");
    expect(getAnalysis).toHaveBeenCalledWith("m1");
  });

  it("renders transcript turns in order with unchanged speaker labels", async () => {
    render(<MeetingDetail meetingId="m1" />);

    const items = await screen.findAllByRole("listitem");
    const texts = items.map((item) => item.textContent ?? "");
    expect(texts[0]).toContain("Kişi 1");
    expect(texts[1]).toContain("Kişi 2");
    expect(texts.some((text) => text.includes("Bilinmeyen"))).toBe(true);
  });

  it("seeks the audio player when a transcript timestamp is clicked", async () => {
    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("Tamam");

    const seekButtons = screen.getAllByRole("button", { name: /Sesi 00:12 konumuna getir/i });
    fireEvent.click(seekButtons[0]);

    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio).toBeTruthy();
    expect(audio.currentTime).toBeCloseTo(12.35, 2);
    expect(window.HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });

  it("keeps the transcript usable when analysis is not configured (503)", async () => {
    startAnalysis.mockRejectedValue(new ApiError(503, "provider not configured"));

    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("Merhaba");

    fireEvent.click(screen.getByRole("button", { name: "Toplantıyı Analiz Et" }));

    expect(
      await screen.findByText(
        "Toplantı analizi için uygun bir LLM sağlayıcısı yapılandırılmamış.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("Merhaba")).toBeTruthy();
  });

  it("renders completed analysis sections and analysis timestamp seeks", async () => {
    getAnalysis.mockResolvedValue(COMPLETED_ANALYSIS);

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("Özet")).toBeTruthy();
    expect(screen.getByText("Toplantı özeti.")).toBeTruthy();
    expect(screen.getByText("konu bir")).toBeTruthy();
    expect(screen.getByText("Kararlar")).toBeTruthy();
    expect(screen.getByText("Karar verildi.")).toBeTruthy();
    expect(screen.getByText("Aksiyonlar")).toBeTruthy();
    expect(screen.getByText(/Rapor yazılacak/)).toBeTruthy();
    expect(screen.getByText(/Kişi 2 · cuma/)).toBeTruthy();
    expect(screen.getByText("Önemli Anlar")).toBeTruthy();
    expect(screen.getByText("Önemli an")).toBeTruthy();

    const decisionSeek = screen.getAllByRole("button", {
      name: /Sesi 00:12 konumuna getir/i,
    })[0];
    fireEvent.click(decisionSeek);
    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio.currentTime).toBeCloseTo(12.35, 2);
  });

  it("shows the failure reason for a failed meeting", async () => {
    getMeeting.mockResolvedValue({
      ...COMPLETED_MEETING,
      status: "failed",
      processing_error: "whisper.cpp model not found",
      has_transcript: false,
    });

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText(/whisper\.cpp model not found/)).toBeTruthy();
  });

  it("polls while processing and stops once completed", async () => {
    vi.useFakeTimers();
    try {
      getMeeting
        .mockResolvedValueOnce({ ...COMPLETED_MEETING, status: "processing", has_transcript: false })
        .mockResolvedValue(COMPLETED_MEETING);

      render(<MeetingDetail meetingId="m1" />);
      await act(async () => {
        await Promise.resolve();
      });
      expect(getMeeting).toHaveBeenCalledTimes(1);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(2100);
      });
      expect(getMeeting).toHaveBeenCalledTimes(2);
      expect(getTranscript).toHaveBeenCalled();

      // Completed: no further polling even after the interval passes again.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(4200);
      });
      expect(getMeeting).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });
});
