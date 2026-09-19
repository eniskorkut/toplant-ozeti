import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PcmCapture, Resampler, floatToInt16 } from "@/lib/pcm-capture";
import { RollingSpeakerTracker } from "@/lib/rolling-speaker";
import {
  ScribeRealtimeClient,
  arrayBufferToBase64,
  buildRealtimeUrl,
  parseWords,
} from "@/lib/scribe-realtime";

describe("Resampler", () => {
  it("downsamples 48 kHz input to roughly one third of the samples", () => {
    const resampler = new Resampler(48000, 16000);
    let output = 0;
    for (let frame = 0; frame < 375; frame += 1) {
      output += resampler.push(new Float32Array(128)).length; // 48000 samples
    }
    expect(output).toBeGreaterThan(15_800);
    expect(output).toBeLessThan(16_200);
  });

  it("passes 16 kHz input through without dropping samples", () => {
    const resampler = new Resampler(16000, 16000);
    const output = resampler.push(new Float32Array(1600));
    expect(output.length).toBeGreaterThanOrEqual(1599);
    expect(output.length).toBeLessThanOrEqual(1600);
  });
});

describe("floatToInt16", () => {
  it("clamps and converts", () => {
    const pcm = floatToInt16(new Float32Array([0, 1, -1, 2, -2]));
    expect(pcm[0]).toBe(0);
    expect(pcm[1]).toBe(32767);
    expect(pcm[2]).toBe(-32768);
    expect(pcm[3]).toBe(32767);
    expect(pcm[4]).toBe(-32768);
  });
});

describe("scribe realtime helpers", () => {
  it("builds the documented URL with VAD defaults and the single-use token", () => {
    const url = buildRealtimeUrl("sutkn_test");
    expect(url.startsWith("wss://api.elevenlabs.io/v1/speech-to-text/realtime?")).toBe(true);
    const params = new URL(url).searchParams;
    expect(params.get("model_id")).toBe("scribe_v2_realtime");
    expect(params.get("language_code")).toBe("tur");
    expect(params.get("include_timestamps")).toBe("true");
    expect(params.get("commit_strategy")).toBe("vad");
    expect(params.get("vad_silence_threshold_secs")).toBe("1.5");
    expect(params.get("vad_threshold")).toBe("0.4");
    expect(params.get("min_speech_duration_ms")).toBe("100");
    expect(params.get("min_silence_duration_ms")).toBe("100");
    expect(params.get("token")).toBe("sutkn_test");
  });

  it("base64-encodes audio and parses word timestamps safely", () => {
    const buffer = new Uint8Array([0, 1, 2, 253, 254, 255]).buffer;
    const decoded = atob(arrayBufferToBase64(buffer));
    expect([...decoded].map((char) => char.charCodeAt(0))).toEqual([0, 1, 2, 253, 254, 255]);
    expect(parseWords({ words: [{ text: "a", start: 0, end: 1 }, { nope: true }] })).toEqual([
      { text: "a", start: 0, end: 1 },
    ]);
  });
});

class FakeWebSocket {
  static OPEN = 1;
  static instances: FakeWebSocket[] = [];
  url: string;
  readyState = 0;
  sent: string[] = [];
  onopen: ((event: unknown) => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  onclose: ((event: unknown) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.readyState = 3;
    this.onclose?.({});
  }

  emitOpen() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.({});
  }

  emitMessage(payload: unknown) {
    this.onmessage?.({ data: JSON.stringify(payload) });
  }

  emitClose() {
    this.readyState = 3;
    this.onclose?.({});
  }
}

describe("ScribeRealtimeClient", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  function makeClient() {
    const partials: string[] = [];
    const committed: { text: string; words: unknown[] }[] = [];
    const statuses: string[] = [];
    const client = new ScribeRealtimeClient({
      onPartial: (text) => partials.push(text),
      onCommitted: (text, words) => committed.push({ text, words }),
      onStatus: (status) => statuses.push(status),
    });
    return { client, partials, committed, statuses };
  }

  it("connects, sends PCM chunks and parses partial/committed messages", async () => {
    const { client, partials, committed, statuses } = makeClient();
    client.open(async () => "sutkn_x");
    await vi.advanceTimersByTimeAsync(0);
    const socket = FakeWebSocket.instances[0];
    socket.emitOpen();
    expect(statuses).toEqual(["connecting", "connected"]);

    client.sendAudio(new Int16Array([1, 2]).buffer);
    const sent = JSON.parse(socket.sent[0]);
    expect(sent.message_type).toBe("input_audio_chunk");
    expect(sent.sample_rate).toBe(16000);
    expect(typeof sent.audio_base_64).toBe("string");

    socket.emitMessage({ message_type: "partial_transcript", text: "merhaba" });
    socket.emitMessage({
      message_type: "committed_transcript_with_timestamps",
      text: "merhaba dünya",
      words: [{ text: "merhaba", start: 0.1, end: 0.5 }],
    });
    expect(partials).toEqual(["merhaba"]);
    expect(committed[0].text).toBe("merhaba dünya");
    expect(committed[0].words).toEqual([{ text: "merhaba", start: 0.1, end: 0.5 }]);

    client.commit();
    expect(JSON.parse(socket.sent[socket.sent.length - 1])).toEqual({
      message_type: "commit",
    });
  });

  it("reconnects a bounded number of times, then reports failure", async () => {
    const { client, statuses } = makeClient();
    client.open(async () => "sutkn_x");
    await vi.advanceTimersByTimeAsync(0);
    FakeWebSocket.instances[0].emitOpen();
    FakeWebSocket.instances[0].emitClose();

    await vi.advanceTimersByTimeAsync(600);
    expect(FakeWebSocket.instances).toHaveLength(2);
    FakeWebSocket.instances[1].emitClose();
    await vi.advanceTimersByTimeAsync(1200);
    expect(FakeWebSocket.instances).toHaveLength(3);
    FakeWebSocket.instances[2].emitClose();
    await vi.advanceTimersByTimeAsync(5000);

    expect(FakeWebSocket.instances).toHaveLength(3);
    expect(statuses[statuses.length - 1]).toBe("failed");
  });

  it("requests a fresh single-use token for every connection attempt", async () => {
    const tokens = ["sutkn_a", "sutkn_b", "sutkn_c"];
    const getToken = vi.fn(async () => tokens[getToken.mock.calls.length - 1] ?? "sutkn_extra");
    const { client } = makeClient();

    client.open(getToken);
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).toContain("token=sutkn_a");
    FakeWebSocket.instances[0].emitOpen();

    FakeWebSocket.instances[0].emitClose();
    await vi.advanceTimersByTimeAsync(600);

    expect(getToken).toHaveBeenCalledTimes(2);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(FakeWebSocket.instances[1].url).toContain("token=sutkn_b");
    expect(FakeWebSocket.instances[1].url).not.toContain("token=sutkn_a");
  });

  it("reconnects after more than the 15 minute token lifetime with a new token", async () => {
    const tokens = ["sutkn_old", "sutkn_new"];
    const getToken = vi.fn(async () => tokens[Math.min(getToken.mock.calls.length - 1, 1)]);
    const { client, statuses } = makeClient();

    client.open(getToken);
    await vi.advanceTimersByTimeAsync(0);
    expect(FakeWebSocket.instances[0].url).toContain("token=sutkn_old");
    FakeWebSocket.instances[0].emitOpen();
    expect(statuses).toContain("connected");

    // Time passes beyond the token lifetime, then the socket fails.
    await vi.advanceTimersByTimeAsync(16 * 60 * 1000);
    FakeWebSocket.instances[0].emitClose();
    await vi.advanceTimersByTimeAsync(600);

    expect(getToken).toHaveBeenCalledTimes(2);
    expect(FakeWebSocket.instances[1].url).toContain("token=sutkn_new");
    expect(FakeWebSocket.instances[1].url).not.toContain("token=sutkn_old");
    FakeWebSocket.instances[1].emitOpen();
    expect(statuses[statuses.length - 1]).toBe("connected");
  });

  it("never throws when sending without an open socket", async () => {
    const { client } = makeClient();
    client.open(async () => "sutkn_x");
    await vi.advanceTimersByTimeAsync(0);
    FakeWebSocket.instances[0].readyState = 3;
    expect(() => client.sendAudio(new Int16Array([1]).buffer)).not.toThrow();
  });
});

describe("PcmCapture cadence", () => {
  it("emits ~100 ms chunks continuously at 48 kHz input", () => {
    const chunks: { bytes: number; startSeconds: number }[] = [];
    const capture = new PcmCapture({} as MediaStream, {
      onChunk: (pcm, startSeconds) => chunks.push({ bytes: pcm.byteLength, startSeconds }),
    });

    // 1 second of 48 kHz audio in 128-sample AudioWorklet quanta.
    for (let frame = 0; frame < 375; frame += 1) {
      capture.handleFrame(new Float32Array(128));
    }

    expect(chunks.length).toBeGreaterThanOrEqual(9);
    expect(chunks.length).toBeLessThanOrEqual(11);
    expect(chunks[0].bytes).toBe(3200);
    expect(chunks[1].startSeconds).toBeCloseTo(0.1, 3);
    expect(capture.streamedSeconds).toBeGreaterThan(0.9);
    const stats = capture.stats;
    expect(stats.chunks).toBe(chunks.length);
    expect(stats.gapCount).toBe(0);
    expect(capture.stats.averageIntervalMs).not.toBeNull();
  });

  it("does not batch audio into multi-second payloads", () => {
    const sizes: number[] = [];
    const capture = new PcmCapture({} as MediaStream, {
      onChunk: (pcm) => sizes.push(pcm.byteLength),
    });
    for (let frame = 0; frame < 375 * 3; frame += 1) {
      capture.handleFrame(new Float32Array(128));
    }
    expect(Math.max(...sizes)).toBe(3200);
    expect(sizes.length).toBeGreaterThan(25);
  });
});

describe("RollingSpeakerTracker", () => {
  function pcm(seconds: number): Int16Array {
    return new Int16Array(16_000 * seconds);
  }

  it("sends non-growing overlapping windows with increasing sequence numbers", async () => {
    const sent: { startSeconds: number; endSeconds: number; sequence: number }[] = [];
    const tracker = new RollingSpeakerTracker({
      liveSessionId: "live-1",
      speakerCount: null,
      send: async (request) => {
        sent.push({
          startSeconds: request.startSeconds,
          endSeconds: request.endSeconds,
          sequence: request.sequence,
        });
        return {
          sequence: request.sequence,
          window: [request.startSeconds, request.endSeconds],
          assignments: [],
          new_speakers: [],
          provider_speakers: 1,
          latency_seconds: 0.4,
          rolling_seconds: request.endSeconds,
          label_switches: 0,
        };
      },
      onResult: () => undefined,
    });

    for (let second = 0; second < 4; second += 1) {
      tracker.push(pcm(1), second);
    }
    await Promise.resolve();
    await Promise.resolve();
    expect(sent).toEqual([{ startSeconds: 0, endSeconds: 4, sequence: 1 }]);

    for (let second = 4; second < 8; second += 1) {
      tracker.push(pcm(1), second);
    }
    await Promise.resolve();
    await Promise.resolve();
    expect(sent).toEqual([
      { startSeconds: 0, endSeconds: 4, sequence: 1 },
      { startSeconds: 3, endSeconds: 7, sequence: 2 },
    ]);
    expect(tracker.requestCount).toBe(2);
    expect(tracker.uploadedSeconds).toBe(8);
  });

  it("retries the same sequence once after a failure, on a delay", async () => {
    vi.useFakeTimers();
    try {
      const sequences: number[] = [];
      let calls = 0;
      const tracker = new RollingSpeakerTracker({
        liveSessionId: "live-1",
        speakerCount: null,
        send: async (request) => {
          calls += 1;
          sequences.push(request.sequence);
          if (calls === 1) throw new Error("provider down");
          return {
            sequence: request.sequence,
            window: [request.startSeconds, request.endSeconds],
            assignments: [],
            new_speakers: [],
            provider_speakers: 1,
            latency_seconds: 0.4,
            rolling_seconds: request.endSeconds,
            label_switches: 0,
          };
        },
        onResult: () => undefined,
        onFailure: () => undefined,
      });

      for (let second = 0; second < 4; second += 1) tracker.push(pcm(1), second);
      await vi.advanceTimersByTimeAsync(0);
      expect(sequences).toEqual([1]);

      // No immediate microtask retry loop: the retry waits for the backoff.
      await vi.advanceTimersByTimeAsync(1600);
      expect(sequences).toEqual([1, 1]);
      expect(tracker.requestCount).toBe(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("skips a window after repeated failures instead of retrying forever", async () => {
    vi.useFakeTimers();
    try {
      const sequences: number[] = [];
      const tracker = new RollingSpeakerTracker({
        liveSessionId: "live-1",
        speakerCount: null,
        send: async (request) => {
          sequences.push(request.sequence);
          throw new Error("provider down");
        },
        onResult: () => undefined,
        onFailure: () => undefined,
      });

      for (let second = 0; second < 7; second += 1) tracker.push(pcm(1), second);
      await vi.advanceTimersByTimeAsync(0);
      expect(sequences).toEqual([1]);
      await vi.advanceTimersByTimeAsync(1600); // second failure -> window skipped
      // The next window uses a fresh sequence immediately; window 1 is never
      // retried forever.
      expect(sequences).toEqual([1, 1, 2]);
      expect(tracker.requestCount).toBeGreaterThanOrEqual(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops cleanly", () => {
    const tracker = new RollingSpeakerTracker({
      liveSessionId: "live-1",
      speakerCount: null,
      send: async (request) => ({
        sequence: request.sequence,
        window: [request.startSeconds, request.endSeconds],
        assignments: [],
        new_speakers: [],
        provider_speakers: 1,
        latency_seconds: 0.4,
        rolling_seconds: request.endSeconds,
        label_switches: 0,
      }),
      onResult: () => undefined,
    });
    tracker.stop();
    for (let second = 0; second < 6; second += 1) tracker.push(pcm(1), second);
    expect(tracker.requestCount).toBe(0);
  });
});
