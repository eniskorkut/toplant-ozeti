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
const getTranscriptionProviders = vi.fn();
const getElevenLabsUsage = vi.fn();
const createLiveSession = vi.fn();
const deleteLiveSession = vi.fn();
const getRealtimeToken = vi.fn();
const sendSpeakerWindow = vi.fn();
const setLiveSpeakerAlias = vi.fn();
const clearLiveSpeakerAlias = vi.fn();

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    uploadRecording: (...args: unknown[]) => uploadRecording(...args),
    processMeeting: (...args: unknown[]) => processMeeting(...args),
    getMeeting: (...args: unknown[]) => getMeeting(...args),
    getTranscriptionProviders: (...args: unknown[]) => getTranscriptionProviders(...args),
    getElevenLabsUsage: (...args: unknown[]) => getElevenLabsUsage(...args),
    createLiveSession: (...args: unknown[]) => createLiveSession(...args),
    deleteLiveSession: (...args: unknown[]) => deleteLiveSession(...args),
    getRealtimeToken: (...args: unknown[]) => getRealtimeToken(...args),
    sendSpeakerWindow: (...args: unknown[]) => sendSpeakerWindow(...args),
    setLiveSpeakerAlias: (...args: unknown[]) => setLiveSpeakerAlias(...args),
    clearLiveSpeakerAlias: (...args: unknown[]) => clearLiveSpeakerAlias(...args),
  };
});

const scribeCallbacks: Array<{
  onPartial: (text: string) => void;
  onCommitted: (text: string, words: unknown[]) => void;
  onStatus: (status: string) => void;
  onTimeline?: (event: string, atMs: number) => void;
}> = [];
const scribeTokenProviders: Array<() => Promise<string>> = [];
vi.mock("@/lib/scribe-realtime", () => ({
  ScribeRealtimeClient: class {
    private callbacks: (typeof scribeCallbacks)[number];
    constructor(callbacks: (typeof scribeCallbacks)[number]) {
      this.callbacks = callbacks;
      scribeCallbacks.push(callbacks);
    }
    open(getToken: () => Promise<string>) {
      scribeTokenProviders.push(getToken);
      this.callbacks.onStatus("connected");
      this.callbacks.onTimeline?.("connected", performance.now());
    }
    sendAudio() {}
    commit() {}
    close() {}
  },
}));

const pcmOptions: {
  current: null | {
    onChunk: (pcm: ArrayBuffer, startSeconds: number) => void;
    onError?: (error: Error) => void;
  };
} = { current: null };

vi.mock("@/lib/pcm-capture", () => ({
  PcmCapture: class {
    constructor(_stream: unknown, options: (typeof pcmOptions)["current"]) {
      pcmOptions.current = options;
    }
    async start() {
      return 48000;
    }
    stop() {}
  },
}));

vi.mock("@/lib/audio-recorder", () => ({
  MeetingRecorder: class {
    get audioStream() {
      return { id: "stream" };
    }
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

function capabilities(elevenlabsAvailable: boolean) {
  return {
    default: "local",
    providers: [
      { id: "local", available: true, cloud: false, label: "Yerel" },
      { id: "elevenlabs", available: elevenlabsAvailable, cloud: true, label: "ElevenLabs" },
    ],
  };
}

beforeEach(() => {
  getTranscriptionProviders.mockResolvedValue(capabilities(false));
  getElevenLabsUsage.mockResolvedValue({ available: false, reason: "usage_scope_unavailable" });
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

beforeEach(() => {
  scribeCallbacks.length = 0;
  scribeTokenProviders.length = 0;
  pcmOptions.current = null;
  createLiveSession.mockResolvedValue({ live_session_id: "live-1" });
  deleteLiveSession.mockResolvedValue(undefined);
  getRealtimeToken.mockResolvedValue({ token: "sutkn_test" });
  sendSpeakerWindow.mockResolvedValue({
    sequence: 1,
    window: [0, 4],
    assignments: [],
    new_speakers: [],
    provider_speakers: 0,
    latency_seconds: 0.5,
    rolling_seconds: 4,
    label_switches: 0,
  });
  setLiveSpeakerAlias.mockResolvedValue({ canonical_speaker: "Kişi 1", display_name: "Ahmet" });
  clearLiveSpeakerAlias.mockResolvedValue(undefined);
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
    expect(processMeeting).toHaveBeenCalledWith("m-new", {
      speakerCount: 2,
      transcriptionProvider: "local",
    });
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

    expect(processMeeting).toHaveBeenCalledWith("m-new", {
      speakerCount: null,
      transcriptionProvider: "local",
    });
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

    expect(processMeeting).toHaveBeenCalledWith("m-new", {
      speakerCount: 8,
      transcriptionProvider: "local",
    });
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


describe("RecordingPanel provider selection", () => {
  it("defaults to Yerel and fetches capabilities from the backend", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });

    expect(getTranscriptionProviders).toHaveBeenCalled();
    const local = screen.getByRole("radio", { name: /Yerel/ });
    expect(local).toBeChecked();
    expect(screen.getByText("Ses cihazınızdan dışarı gönderilmez.")).toBeTruthy();
  });

  it("disables ElevenLabs and shows the safe message when unavailable", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });

    const elevenlabs = screen.getByRole("radio", { name: /ElevenLabs/ });
    expect(elevenlabs).toBeDisabled();
    expect(screen.getByText("ElevenLabs API yapılandırılmamış.")).toBeTruthy();
  });

  it("requires cloud acknowledgment before recording and keeps it for queueing", async () => {
    getTranscriptionProviders.mockResolvedValue(capabilities(true));
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("radio", { name: /ElevenLabs/ }));
    });
    const startRecording = screen.getByRole("button", { name: "Kaydı Başlat" });
    expect(startRecording).toBeDisabled();
    expect(screen.getByText(/ElevenLabs'a gönderileceğini/)).toBeTruthy();
    expect(screen.getByText(/beklenen azami sayıdır/)).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole("checkbox"));
    });
    expect(startRecording).not.toBeDisabled();

    await recordAndUpload();
    await queueProcessing();

    expect(processMeeting).toHaveBeenCalledWith("m-new", {
      speakerCount: null,
      transcriptionProvider: "elevenlabs",
    });
  });

  it("does not require consent for the local provider", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.getByRole("button", { name: "Kaydı Başlat" })).not.toBeDisabled();
    expect(screen.queryByRole("checkbox")).toBeNull();
  });
});


describe("RecordingPanel ElevenLabs usage display", () => {
  it("shows a compact usage summary when the API is available", async () => {
    getTranscriptionProviders.mockResolvedValue(capabilities(true));
    getElevenLabsUsage.mockResolvedValue({
      available: true,
      tier: "free",
      usage: 1200,
      limit: 10000,
      remaining: 8800,
      reset_at: "2026-10-09T08:53:20+00:00",
    });

    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("radio", { name: /ElevenLabs/ }));
    });

    expect(await screen.findByText(/Plan:/)).toBeTruthy();
    expect(screen.getByText(/Kullanım:/)).toBeTruthy();
    expect(screen.getByText("8800")).toBeTruthy();
    expect(screen.getByText(/Yenilenme:/)).toBeTruthy();
  });

  it("shows the panel guidance when usage scope is unavailable", async () => {
    getTranscriptionProviders.mockResolvedValue(capabilities(true));

    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("radio", { name: /ElevenLabs/ }));
    });

    expect(
      await screen.findByText(/Developers → Analytics → Usage/),
    ).toBeTruthy();
  });

  it("keeps provider selection usable when the usage lookup fails", async () => {
    getTranscriptionProviders.mockResolvedValue(capabilities(true));
    getElevenLabsUsage.mockRejectedValue(new Error("usage request failed"));

    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("radio", { name: /ElevenLabs/ }));
    });

    expect(await screen.findByText(/Developers → Analytics → Usage/)).toBeTruthy();
    expect(screen.getByRole("checkbox")).toBeTruthy();
  });
});

async function selectLiveProvider() {
  await act(async () => {
    fireEvent.click(screen.getByRole("radio", { name: /ElevenLabs/ }));
  });
  await act(async () => {
    fireEvent.click(screen.getByRole("checkbox"));
  });
}

async function startLiveRecording() {
  await act(async () => {
    fireEvent.click(screen.getByRole("button", { name: "Kaydı Başlat" }));
  });
}

function emitChunks(seconds: number) {
  for (let second = 0; second < seconds; second += 1) {
    pcmOptions.current?.onChunk(new Int16Array(16_000).buffer, second);
  }
}

function rollingResult(overrides: Record<string, unknown> = {}) {
  return {
    sequence: 1,
    window: [0, 4],
    assignments: [
      {
        canonical_speaker: "Kişi 1",
        is_new: true,
        confidence: null,
        start: 0,
        end: 4,
        speech_seconds: 2,
      },
    ],
    new_speakers: ["Kişi 1"],
    provider_speakers: 1,
    latency_seconds: 0.6,
    rolling_seconds: 4,
    label_switches: 0,
    ...overrides,
  };
}

describe("RecordingPanel live ElevenLabs mode", () => {
  beforeEach(() => {
    getTranscriptionProviders.mockResolvedValue(capabilities(true));
  });

  it("shows partials, replaces them with committed text and never duplicates", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    expect(scribeCallbacks).toHaveLength(1);
    expect(screen.getByText("Canlı transkript")).toBeTruthy();

    await act(async () => {
      scribeCallbacks[0].onPartial("Yarın nereye");
    });
    // Partial text is visible immediately: no rolling window, no speaker
    // attribution, no committed transcript, no MediaRecorder chunk required.
    expect(screen.getByText("Yarın nereye")).toBeTruthy();
    expect(screen.getByText("Konuşmacı belirleniyor")).toBeTruthy();
    expect(sendSpeakerWindow).not.toHaveBeenCalled();

    await act(async () => {
      scribeCallbacks[0].onCommitted("Yarın nereye gideceğiz?", [
        { text: "Yarın", start: 1, end: 2 },
      ]);
    });
    expect(screen.queryByText("Yarın nereye")).toBeNull();
    expect(screen.getAllByText("Yarın nereye gideceğiz?")).toHaveLength(1);

    await act(async () => {
      scribeCallbacks[0].onPartial("Kentpark'a");
    });
    await act(async () => {
      scribeCallbacks[0].onCommitted("Kentpark'a gideriz.", [
        { text: "Kentpark'a", start: 3, end: 4 },
      ]);
    });
    expect(screen.queryByText("Kentpark'a")).toBeNull();
    expect(screen.getByText("Kentpark'a gideriz.")).toBeTruthy();
  });

  it("asks the realtime client for a fresh token through the backend provider", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    expect(scribeTokenProviders).toHaveLength(1);
    await act(async () => {
      await scribeTokenProviders[0]();
    });
    expect(getRealtimeToken).toHaveBeenCalledTimes(1);
    // A reconnect in the client would call the same provider again; the panel
    // never caches a token itself.
    await act(async () => {
      await scribeTokenProviders[0]();
    });
    expect(getRealtimeToken).toHaveBeenCalledTimes(2);
  });

  it("keeps committed text across a realtime reconnect", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    await act(async () => {
      scribeCallbacks[0].onCommitted("bağlantı kopmadan önce", [
        { text: "bağlantı", start: 1, end: 2 },
      ]);
    });

    await act(async () => {
      scribeCallbacks[0].onStatus("reconnecting");
    });
    expect(screen.getByText("bağlantı kopmadan önce")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Kaydı Durdur" })).toBeTruthy();

    await act(async () => {
      scribeCallbacks[0].onStatus("connected");
    });
    expect(screen.getByText("bağlantı kopmadan önce")).toBeTruthy();
  });

  it("changes only the speaker label when diarization arrives after the text", async () => {
    sendSpeakerWindow.mockResolvedValue(rollingResult());
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    await act(async () => {
      scribeCallbacks[0].onCommitted("metin aynı kalmalı", [
        { text: "metin", start: 1, end: 2 },
      ]);
    });
    const before = screen.getByText("metin aynı kalmalı").textContent;

    await act(async () => {
      emitChunks(4);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText("metin aynı kalmalı").textContent).toBe(before);
    expect(screen.getByText("Kişi 1")).toBeTruthy();
    expect(screen.queryByText("Konuşmacı belirleniyor")).toBeNull();
  });

  it("attaches Kişi labels after the rolling window result arrives", async () => {
    sendSpeakerWindow.mockResolvedValue(rollingResult());
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    await act(async () => {
      scribeCallbacks[0].onCommitted("Yarın nereye gideceğiz?", [
        { text: "Yarın", start: 1, end: 2 },
      ]);
    });
    expect(screen.getByText("Konuşmacı belirleniyor")).toBeTruthy();

    await act(async () => {
      emitChunks(4);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(sendSpeakerWindow).toHaveBeenCalledWith(
      "live-1",
      expect.objectContaining({ sequence: 1, startSeconds: 0, endSeconds: 4 }),
    );
    expect(screen.queryByText("Konuşmacı belirleniyor")).toBeNull();
    expect(screen.getByText("Kişi 1")).toBeTruthy();
  });

  it("renames Kişi 1 to Ahmet and applies it to existing and future lines", async () => {
    sendSpeakerWindow.mockResolvedValue(rollingResult());
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    await act(async () => {
      scribeCallbacks[0].onCommitted("ilk cümle", [{ text: "ilk", start: 1, end: 2 }]);
      emitChunks(4);
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByText("Kişi 1")).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Kişi 1 · Konuşmacı adını düzenle/ }));
    });
    const input = screen.getByLabelText("Ad");
    fireEvent.change(input, { target: { value: "Ahmet" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    });

    expect(setLiveSpeakerAlias).toHaveBeenCalledWith("live-1", "Kişi 1", "Ahmet");
    expect(screen.getByText("Ahmet")).toBeTruthy();
    expect(screen.queryByText("Kişi 1")).toBeNull();

    // Future lines with the same canonical label display the alias too.
    await act(async () => {
      scribeCallbacks[0].onCommitted("ikinci cümle", [{ text: "ikinci", start: 2, end: 3 }]);
    });
    expect(screen.getAllByText("Ahmet").length).toBeGreaterThanOrEqual(2);
  });

  it("resets an alias back to the canonical label", async () => {
    sendSpeakerWindow.mockResolvedValue(rollingResult());
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();
    await act(async () => {
      scribeCallbacks[0].onCommitted("ilk cümle", [{ text: "ilk", start: 1, end: 2 }]);
      emitChunks(4);
      await Promise.resolve();
      await Promise.resolve();
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Kişi 1 · Konuşmacı adını düzenle/ }));
    });
    fireEvent.change(screen.getByLabelText("Ad"), { target: { value: "Ahmet" } });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydet" }));
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /Ahmet · Konuşmacı adını düzenle/ }));
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: /sıfırla/ }));
    });

    expect(clearLiveSpeakerAlias).toHaveBeenCalledWith("live-1", "Kişi 1");
    expect(screen.getByText("Kişi 1")).toBeTruthy();
    expect(screen.queryByText("Ahmet")).toBeNull();
  });

  it("keeps recording when the realtime socket fails", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();

    await act(async () => {
      scribeCallbacks[0].onStatus("failed");
    });

    expect(
      screen.getAllByText(/Canlı transkript bağlantısı kesildi — kayıt devam ediyor/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Kaydı Durdur" })).toBeTruthy();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydı Durdur" }));
    });
    expect(uploadRecording).toHaveBeenCalled();
  });

  it("keeps live text and recording when rolling diarization fails", async () => {
    sendSpeakerWindow.mockRejectedValue(new Error("rolling down"));
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();
    await act(async () => {
      scribeCallbacks[0].onCommitted("metin kaybolmasın", [
        { text: "metin", start: 1, end: 2 },
      ]);
      emitChunks(4);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(screen.getByText("metin kaybolmasın")).toBeTruthy();
    expect(
      screen.getAllByText(/Konuşmacı etiketleri şu an alınamıyor/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: "Kaydı Durdur" })).toBeTruthy();
  });

  it("uploads with the live session id and replaces live text with the final state", async () => {
    vi.useFakeTimers();
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await selectLiveProvider();
    await startLiveRecording();
    await act(async () => {
      scribeCallbacks[0].onCommitted("geçici metin", [{ text: "geçici", start: 1, end: 2 }]);
    });

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Kaydı Durdur" }));
    });
    expect(uploadRecording).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ liveSessionId: "live-1" }),
    );

    await queueProcessing();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2100);
    });

    expect(screen.getByText("Transkripsiyon tamamlandı.")).toBeTruthy();
    expect(screen.queryByText("Canlı transkript")).toBeNull();
    expect(screen.queryByText("geçici metin")).toBeNull();
  });

  it("does not start the live pipeline for the local provider", async () => {
    render(<RecordingPanel />);
    await act(async () => {
      await Promise.resolve();
    });
    await recordAndUpload();
    expect(createLiveSession).not.toHaveBeenCalled();
    expect(getRealtimeToken).not.toHaveBeenCalled();
  });
});
