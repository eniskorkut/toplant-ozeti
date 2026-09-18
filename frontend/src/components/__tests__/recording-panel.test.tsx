import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RecordingPanel } from "@/components/recording-panel";

const POLL_INTERVAL = 2000;

const start = vi.fn();
const stop = vi.fn();
const dispose = vi.fn();
const uploadRecording = vi.fn();
const processMeeting = vi.fn();
const getMeeting = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    uploadRecording: (...args: unknown[]) => uploadRecording(...args),
    processMeeting: (...args: unknown[]) => processMeeting(...args),
    getMeeting: (...args: unknown[]) => getMeeting(...args),
  };
});

vi.mock("@/lib/audio-recorder", () => ({
  MeetingRecorder: class {
    start(...args: unknown[]) {
      return start(...args);
    }
    stop(...args: unknown[]) {
      return stop(...args);
    }
    dispose(...args: unknown[]) {
      return dispose(...args);
    }
  },
  describeMicrophoneError: (error: unknown) =>
    error instanceof Error ? error.message : "mikrofon hatası",
}));

beforeEach(() => {
  start.mockResolvedValue({
    requestedMimeType: "audio/webm;codecs=opus",
    requestedBitsPerSecond: 64000,
    actualBitsPerSecond: 64000,
    trackSettings: { sampleRate: 48000, channelCount: 1 },
  });
  stop.mockResolvedValue({
    blob: new Blob(["audio"], { type: "audio/webm" }),
    mimeType: "audio/webm;codecs=opus",
    durationMs: 5000,
  });
  uploadRecording.mockResolvedValue({
    recording_id: "m-new",
    meeting_id: "m-new",
    duration_seconds: 5,
    input: { mime_type: "audio/webm", size_bytes: 1000 },
    mp3: { size_bytes: 2000 },
    processing_wav: { size_bytes: 3000, sample_rate: 16000, channels: 1 },
    conversion_ms: 42,
  });
  processMeeting.mockResolvedValue({ meeting_id: "m-new", status: "queued" });
  getMeeting.mockResolvedValue({
    meeting_id: "m-new",
    status: "completed",
    created_at: "2026-09-18T12:00:00",
    duration_seconds: 5,
    requested_speaker_count: null,
    processing_error: null,
    has_transcript: true,
  });
  // jsdom has no object-URL implementation; add only the missing statics.
  URL.createObjectURL = vi.fn(() => "blob:preview");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  vi.clearAllMocks();
  vi.useRealTimers();
});



async function recordAndUpload() {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Kaydı Başlat" }));
  });
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Kaydı Durdur" }));
  });
}

async function queueProcessing(speakerValue?: string) {
  if (speakerValue) {
    fireEvent.change(screen.getByLabelText("Konuşmacı sayısı"), {
      target: { value: speakerValue },
    });
  }
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Transkripsiyonu Başlat" }));
  });
}

function meetingWithStatus(status: string) {
  return {
    meeting_id: "m-new",
    status,
    created_at: "2026-09-18T12:00:00",
    duration_seconds: 5,
    requested_speaker_count: null,
    processing_error: status === "failed" ? "işleme hatası" : null,
    has_transcript: status === "completed",
  };
}

describe("RecordingPanel", () => {
  it("runs record -> upload -> process and reports completion", async () => {
    vi.useFakeTimers();
    render(<RecordingPanel />);

    // record
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydı Başlat" }));
    });
    expect(start).toHaveBeenCalled();
    expect(screen.getByText("Kayıt sürüyor")).toBeTruthy();

    // stop + upload
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydı Durdur" }));
    });
    expect(uploadRecording).toHaveBeenCalled();
    expect(screen.getByLabelText("Konuşmacı sayısı")).toBeTruthy();

    // choose a known speaker count and queue processing
    await act(async () => {
      fireEvent.change(screen.getByLabelText("Konuşmacı sayısı"), { target: { value: "2" } });
      fireEvent.click(screen.getByRole("button", { name: "Transkripsiyonu Başlat" }));
    });
    expect(processMeeting).toHaveBeenCalledWith("m-new", 2);
    expect(screen.getByText(/Ses yazıya dönüştürülüyor/)).toBeTruthy();

    // polling completes
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100);
    });
    expect(getMeeting).toHaveBeenCalledWith("m-new");
    expect(screen.getByText("Transkripsiyon tamamlandı.")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Toplantıyı aç" })).toHaveAttribute(
      "href",
      "/meetings/m-new",
    );

    // no further polling after completion
    const callsAfterCompletion = getMeeting.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(6000);
    });
    expect(getMeeting.mock.calls.length).toBe(callsAfterCompletion);
  });

  it("sends null for automatic speaker count", async () => {
    render(<RecordingPanel />);
    await recordAndUpload();
    await queueProcessing();

    expect(processMeeting).toHaveBeenCalledWith("m-new", null);
  });

  it("renders a backend processing failure", async () => {
    vi.useFakeTimers();
    getMeeting.mockResolvedValue({
      meeting_id: "m-new",
      status: "failed",
      created_at: "2026-09-18T12:00:00",
      duration_seconds: 5,
      requested_speaker_count: null,
      processing_error: "whisper.cpp model not found",
      has_transcript: false,
    });

    render(<RecordingPanel />);
    await recordAndUpload();
    await queueProcessing();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL + 50);
    });

    expect(screen.getByRole("alert")).toHaveTextContent("whisper.cpp model not found");
  });

  it("keeps polling through repeated processing responses until completed", async () => {
    vi.useFakeTimers();
    getMeeting
      .mockResolvedValueOnce(meetingWithStatus("processing"))
      .mockResolvedValueOnce(meetingWithStatus("processing"))
      .mockResolvedValueOnce(meetingWithStatus("processing"))
      .mockResolvedValueOnce(meetingWithStatus("completed"));

    render(<RecordingPanel />);
    await recordAndUpload();
    await queueProcessing();

    // poll 1
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL + 50);
    });
    expect(getMeeting).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/Ses yazıya dönüştürülüyor/)).toBeTruthy();

    // polls 2 and 3 keep happening even though the UI already says "processing"
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL + 50);
    });
    expect(getMeeting).toHaveBeenCalledTimes(2);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL + 50);
    });
    expect(getMeeting).toHaveBeenCalledTimes(3);

    // poll 4 completes the job and polling stops for good
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL + 50);
    });
    expect(getMeeting).toHaveBeenCalledTimes(4);
    expect(screen.getByText("Transkripsiyon tamamlandı.")).toBeTruthy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL * 5);
    });
    expect(getMeeting).toHaveBeenCalledTimes(4);
  });

  it("polls through queued -> processing -> completed", async () => {
    vi.useFakeTimers();
    getMeeting
      .mockResolvedValueOnce(meetingWithStatus("queued"))
      .mockResolvedValueOnce(meetingWithStatus("processing"))
      .mockResolvedValueOnce(meetingWithStatus("completed"));

    render(<RecordingPanel />);
    await recordAndUpload();
    await queueProcessing();

    for (let expected = 1; expected <= 3; expected += 1) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL + 50);
      });
      expect(getMeeting).toHaveBeenCalledTimes(expected);
    }

    expect(screen.getByText("Transkripsiyon tamamlandı.")).toBeTruthy();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(POLL_INTERVAL * 5);
    });
    expect(getMeeting).toHaveBeenCalledTimes(3);
  });

  it("sends a known speaker count above 5 (8)", async () => {
    render(<RecordingPanel />);
    await recordAndUpload();
    await queueProcessing("8");

    expect(processMeeting).toHaveBeenCalledWith("m-new", 8);
  });

  it("shows a microphone error without uploading", async () => {
    start.mockRejectedValue(new Error("Mikrofon izni verilmedi"));

    render(<RecordingPanel />);
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydı Başlat" }));
    });

    expect(screen.getByRole("alert")).toHaveTextContent("Mikrofon izni verilmedi");
    expect(uploadRecording).not.toHaveBeenCalled();
  });
});
