import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "@/lib/api";
import type { MeetingAnalysis, MeetingStatus, Transcript } from "@/lib/api";
import { MeetingDetail } from "@/components/meeting-detail";

const getMeeting = vi.fn();
const getTranscript = vi.fn();
const getAnalysis = vi.fn();
const startAnalysis = vi.fn();
const processMeeting = vi.fn();
const getMeetingSpeakers = vi.fn();
const setMeetingSpeakerAlias = vi.fn();
const clearMeetingSpeakerAlias = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getMeeting: (...args: unknown[]) => getMeeting(...args),
    getTranscript: (...args: unknown[]) => getTranscript(...args),
    getAnalysis: (...args: unknown[]) => getAnalysis(...args),
    startAnalysis: (...args: unknown[]) => startAnalysis(...args),
    processMeeting: (...args: unknown[]) => processMeeting(...args),
    getMeetingSpeakers: (...args: unknown[]) => getMeetingSpeakers(...args),
    setMeetingSpeakerAlias: (...args: unknown[]) => setMeetingSpeakerAlias(...args),
    clearMeetingSpeakerAlias: (...args: unknown[]) => clearMeetingSpeakerAlias(...args),
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
  transcription_provider: null,
  transcription_model: null,
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
  key_points: [
    { text: "Cuma günü yayın hedeflendi.", source_turn_ordinals: [1], timestamp_seconds: 12.35 },
  ],
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
  getMeetingSpeakers.mockResolvedValue({
    meeting_id: "m1",
    speakers: ["Kişi 1", "Kişi 2"],
    aliases: {},
  });
  setMeetingSpeakerAlias.mockResolvedValue({
    canonical_speaker: "Kişi 1",
    display_name: "Ahmet",
  });
  clearMeetingSpeakerAlias.mockResolvedValue(undefined);
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

  it("shows a human header with the raw id only as secondary metadata", async () => {
    render(<MeetingDetail meetingId="m1" />);

    const heading = await screen.findByRole("heading", { level: 1 });
    expect(heading).toHaveTextContent(/Toplantı ·/);
    expect(heading.textContent).not.toContain("m1");
    expect(screen.getByRole("button", { name: "Toplantı kimliğini kopyala" })).toBeTruthy();
    expect(screen.getByTitle("m1")).toBeTruthy();
  });

  it("renders one speaker badge per speaker change but keeps every turn labelled", async () => {
    getTranscript.mockResolvedValue({
      ...TRANSCRIPT,
      turns: [
        { ordinal: 0, speaker: "Kişi 1", start_seconds: 0, end_seconds: 4, text: "Merhaba" },
        { ordinal: 1, speaker: "Kişi 1", start_seconds: 4, end_seconds: 7, text: "devam" },
      ],
      unresolved_turns: 0,
    });

    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("devam");

    expect(screen.getAllByText("Kişi 1")).toHaveLength(2); // visible badge + sr-only repeat
    const items = screen.getAllByRole("listitem");
    expect(items[0].textContent).toContain("Kişi 1");
    expect(items[1].textContent).toContain("Kişi 1");
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

    expect(await screen.findByText("Toplantı Özeti")).toBeTruthy();
    expect(screen.getByText("Toplantı özeti.")).toBeTruthy();
    expect(screen.getByText("konu bir")).toBeTruthy();
    expect(screen.getByText("Kararlar")).toBeTruthy();
    expect(screen.getByText("Karar verildi.")).toBeTruthy();
    expect(screen.getByText("Alınacak Aksiyonlar")).toBeTruthy();
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

  it("renders the provider badge for completed meetings", async () => {
    getMeeting.mockResolvedValue({
      ...COMPLETED_MEETING,
      transcription_provider: "elevenlabs",
      transcription_model: "scribe_v2",
    });

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("ElevenLabs · Scribe v2")).toBeTruthy();
  });

  it("renders legacy meetings without provider metadata", async () => {
    getMeeting.mockResolvedValue({ ...COMPLETED_MEETING, transcription_provider: null });

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("Merhaba")).toBeTruthy();
    expect(screen.queryByText(/Yerel · whisper.cpp|ElevenLabs · Scribe v2/)).toBeNull();
  });

  it("offers an explicit local retry for failed meetings", async () => {
    getMeeting.mockResolvedValue({
      ...COMPLETED_MEETING,
      status: "failed",
      processing_error: "ElevenLabs rejected the request (HTTP 401)",
      has_transcript: false,
    });
    processMeeting.mockResolvedValue({
      ...COMPLETED_MEETING,
      status: "queued",
      processing_error: null,
      has_transcript: false,
    });

    render(<MeetingDetail meetingId="m1" />);
    fireEvent.click(await screen.findByRole("button", { name: "Yerel ile yeniden dene" }));

    await waitFor(() =>
      expect(processMeeting).toHaveBeenCalledWith("m1", {
        speakerCount: null,
        transcriptionProvider: "local",
      }),
    );
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

describe("MeetingDetail speaker aliases", () => {
  it("displays meeting-local aliases in the transcript and analysis owner", async () => {
    getMeetingSpeakers.mockResolvedValue({
      meeting_id: "m1",
      speakers: ["Kişi 1", "Kişi 2"],
      aliases: { "Kişi 1": "Ahmet", "Kişi 2": "Ayşe" },
    });
    getAnalysis.mockResolvedValue(COMPLETED_ANALYSIS);

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("Ahmet")).toBeTruthy();
    expect(screen.getByText("Ayşe")).toBeTruthy();
    // Action item owner is canonical "Kişi 2" internally, displayed as "Ayşe".
    expect(await screen.findByText(/Ayşe · cuma/)).toBeTruthy();
  });

  it("renames a speaker for this meeting only and keeps other meetings clean", async () => {
    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("Merhaba");

    fireEvent.click(
      screen.getAllByRole("button", { name: /Kişi 1 · Konuşmacı adını düzenle/ })[0],
    );
    fireEvent.change(screen.getByLabelText("Ad"), { target: { value: "Ahmet" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    });

    expect(setMeetingSpeakerAlias).toHaveBeenCalledWith("m1", "Kişi 1", "Ahmet");
    expect(await screen.findByText("Ahmet")).toBeTruthy();

    // Resetting restores the canonical label through the same dialog.
    fireEvent.click(
      screen.getAllByRole("button", { name: /Ahmet · Konuşmacı adını düzenle/ })[0],
    );
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /sıfırla/ }));
    });
    expect(clearMeetingSpeakerAlias).toHaveBeenCalledWith("m1", "Kişi 1");
    expect(await screen.findByText("Kişi 1")).toBeTruthy();
  });
});

describe("MeetingDetail analysis aliases, key points and copy", () => {
  it("renders grounded key points with clickable timestamps", async () => {
    getAnalysis.mockResolvedValue(COMPLETED_ANALYSIS);

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("Ana Fikirler")).toBeTruthy();
    expect(screen.getByText("Cuma günü yayın hedeflendi.")).toBeTruthy();

    const seek = screen.getAllByRole("button", { name: /Sesi 00:12 konumuna getir/i })[0];
    fireEvent.click(seek);
    const audio = document.querySelector("audio") as HTMLAudioElement;
    expect(audio.currentTime).toBeCloseTo(12.35, 2);
  });

  it("copies deterministic meeting notes with aliases and headings", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    getAnalysis.mockResolvedValue(COMPLETED_ANALYSIS);
    getMeetingSpeakers.mockResolvedValue({
      meeting_id: "m1",
      speakers: ["Kişi 1", "Kişi 2"],
      aliases: { "Kişi 2": "Mehmet" },
    });

    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("Toplantı Özeti");
    fireEvent.click(screen.getByRole("button", { name: "Toplantı Notlarını Kopyala" }));

    await waitFor(() => expect(writeText).toHaveBeenCalled());
    const text = writeText.mock.calls[0][0] as string;
    for (const heading of [
      "TOPLANTI ÖZETİ",
      "ANA FİKİRLER",
      "KARARLAR",
      "ALINACAK AKSİYONLAR",
      "ÖNEMLİ ANLAR",
    ]) {
      expect(text).toContain(heading);
    }
    expect(text).toContain("Mehmet"); // alias resolved in action owner
    expect(text).not.toContain("Kişi 2 ·"); // canonical replaced by alias
    expect(text).toContain("[00:12]");
    expect(await screen.findByText("Kopyalandı")).toBeTruthy();
  });

  it("survives clipboard failure without losing the analysis", async () => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) },
    });
    getAnalysis.mockResolvedValue(COMPLETED_ANALYSIS);

    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("Toplantı Özeti");
    fireEvent.click(screen.getByRole("button", { name: "Toplantı Notlarını Kopyala" }));

    expect(await screen.findByText(/Panoya kopyalanamadı/)).toBeTruthy();
    expect(screen.getByText("Cuma günü yayın hedeflendi.")).toBeTruthy();
  });

  it("offers an explicit refresh after renaming when analysis is completed", async () => {
    getAnalysis.mockResolvedValue(COMPLETED_ANALYSIS);
    startAnalysis.mockResolvedValue({ ...COMPLETED_ANALYSIS, status: "queued" });

    render(<MeetingDetail meetingId="m1" />);
    await screen.findByText("Toplantı Özeti");

    fireEvent.click(
      screen.getAllByRole("button", { name: /Kişi 1 · Konuşmacı adını düzenle/ })[0],
    );
    fireEvent.change(screen.getByLabelText("Ad"), { target: { value: "Ahmet" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    });

    const refresh = await screen.findByRole("button", {
      name: "Analizi yeni konuşmacı adlarıyla yenile",
    });
    await act(async () => {
      fireEvent.click(refresh);
    });
    expect(startAnalysis).toHaveBeenCalledWith("m1", { refresh: true });
  });

  it("renders legacy analyses without key points", async () => {
    getAnalysis.mockResolvedValue({ ...COMPLETED_ANALYSIS, key_points: [] });

    render(<MeetingDetail meetingId="m1" />);

    expect(await screen.findByText("Toplantı Özeti")).toBeTruthy();
    expect(screen.queryByText("Ana Fikirler")).toBeNull();
    expect(screen.getByText("Kararlar")).toBeTruthy();
  });
});
