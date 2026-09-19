/**
 * ElevenLabs Scribe v2 Realtime client (direct WebSocket, no SDK dependency).
 *
 * Auth uses the short-lived single-use token minted by our backend; the permanent
 * API key never reaches the browser. Recording safety: this client never throws
 * into the recorder — failures surface as status events and bounded reconnects.
 */

export const REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime";
export const REALTIME_MODEL = "scribe_v2_realtime";

export type RealtimeWord = { text: string; start: number; end: number };

export type RealtimeStatus =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "failed";

export type RealtimeCallbacks = {
  onPartial: (text: string) => void;
  onCommitted: (text: string, words: RealtimeWord[]) => void;
  onStatus: (status: RealtimeStatus) => void;
};

export type RealtimeOptions = {
  languageCode?: string;
  includeTimestamps?: boolean;
  /** Conservative documented VAD defaults; not tuned against a single meeting. */
  vadSilenceThresholdSecs?: number;
  vadThreshold?: number;
  minSpeechDurationMs?: number;
  minSilenceDurationMs?: number;
  maxReconnects?: number;
};

export function buildRealtimeUrl(token: string, options: RealtimeOptions = {}): string {
  const params = new URLSearchParams({
    model_id: REALTIME_MODEL,
    language_code: options.languageCode ?? "tur",
    include_timestamps: String(options.includeTimestamps ?? true),
    commit_strategy: "vad",
    vad_silence_threshold_secs: String(options.vadSilenceThresholdSecs ?? 1.5),
    vad_threshold: String(options.vadThreshold ?? 0.4),
    min_speech_duration_ms: String(options.minSpeechDurationMs ?? 100),
    min_silence_duration_ms: String(options.minSilenceDurationMs ?? 100),
    token,
  });
  return `${REALTIME_URL}?${params.toString()}`;
}

export class ScribeRealtimeClient {
  private socket: WebSocket | null = null;
  private token: string | null = null;
  private reconnects = 0;
  private closedByUser = false;
  private readonly maxReconnects: number;

  constructor(
    private readonly callbacks: RealtimeCallbacks,
    private readonly options: RealtimeOptions = {},
  ) {
    this.maxReconnects = options.maxReconnects ?? 2;
  }

  open(token: string): void {
    this.token = token;
    this.closedByUser = false;
    this.reconnects = 0;
    this.connect();
  }

  private connect(): void {
    if (!this.token) return;
    this.callbacks.onStatus(this.reconnects === 0 ? "connecting" : "reconnecting");
    let socket: WebSocket;
    try {
      socket = new WebSocket(buildRealtimeUrl(this.token, this.options));
    } catch (error) {
      this.handleFailure(error);
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      this.reconnects = 0;
      this.callbacks.onStatus("connected");
    };
    socket.onmessage = (event: MessageEvent<string>) => {
      this.handleMessage(event.data);
    };
    socket.onerror = () => {
      // onclose always follows; failure handling lives there.
    };
    socket.onclose = () => {
      this.socket = null;
      if (this.closedByUser) {
        this.callbacks.onStatus("disconnected");
        return;
      }
      if (this.reconnects < this.maxReconnects) {
        this.reconnects += 1;
        const delay = 500 * this.reconnects;
        this.callbacks.onStatus("reconnecting");
        window.setTimeout(() => this.connect(), delay);
      } else {
        this.callbacks.onStatus("failed");
      }
    };
  }

  private handleMessage(raw: string): void {
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(raw) as Record<string, unknown>;
    } catch {
      return;
    }
    const type = String(payload.message_type ?? payload.type ?? "");
    if (type === "partial_transcript") {
      const text = String(payload.text ?? "").trim();
      if (text) this.callbacks.onPartial(text);
    } else if (
      type === "committed_transcript" ||
      type === "committed_transcript_with_timestamps"
    ) {
      const text = String(payload.text ?? "").trim();
      if (!text) return;
      this.callbacks.onCommitted(text, parseWords(payload));
    }
  }

  /** Send one 16 kHz mono s16le chunk. A lost socket never throws. */
  sendAudio(pcm: ArrayBuffer): void {
    this.sendMessage({
      message_type: "input_audio_chunk",
      audio_base_64: arrayBufferToBase64(pcm),
      sample_rate: 16_000,
    });
  }

  commit(): void {
    this.sendMessage({ message_type: "commit" });
  }

  private sendMessage(message: Record<string, unknown>): void {
    const socket = this.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    try {
      socket.send(JSON.stringify(message));
    } catch {
      // Never let realtime I/O break the recording.
    }
  }

  private handleFailure(error: unknown): void {
    void error;
    this.callbacks.onStatus("failed");
  }

  close(): void {
    this.closedByUser = true;
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      try {
        socket.close();
      } catch {
        // already closed
      }
    }
    this.callbacks.onStatus("disconnected");
  }
}

export function parseWords(payload: Record<string, unknown>): RealtimeWord[] {
  const entries = payload.words;
  if (!Array.isArray(entries)) return [];
  const words: RealtimeWord[] = [];
  for (const entry of entries) {
    if (!entry || typeof entry !== "object") continue;
    const record = entry as Record<string, unknown>;
    const text = String(record.text ?? "").trim();
    if (!text) continue;
    words.push({
      text,
      start: Number(record.start ?? 0),
      end: Number(record.end ?? 0),
    });
  }
  return words;
}

export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000;
  for (let index = 0; index < bytes.length; index += chunk) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
  }
  return btoa(binary);
}
