import { getPreferredRecordingMimeType } from "@/lib/browser-media";

export const REQUESTED_BITS_PER_SECOND = 64_000;
export const CHUNK_INTERVAL_MS = 1000;

/**
 * Audio-only constraints. Mono is requested for diarization; echo cancellation,
 * noise suppression and auto gain control are preferences, not requirements,
 * because speech models generally prefer the raw signal but browsers may ignore
 * unsupported constraints.
 */
export const AUDIO_CONSTRAINTS: MediaStreamConstraints = {
  audio: {
    channelCount: { ideal: 1 },
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  },
  video: false,
};

export type TrackSettings = {
  sampleRate: number | null;
  channelCount: number | null;
  echoCancellation: boolean | null;
  noiseSuppression: boolean | null;
  autoGainControl: boolean | null;
};

export type RecorderSession = {
  /** MIME type requested from MediaRecorder; null means browser default. */
  requestedMimeType: string | null;
  requestedBitsPerSecond: number;
  /** What MediaRecorder reports it actually uses. */
  actualBitsPerSecond: number | null;
  trackSettings: TrackSettings;
};

export type RecordingResult = {
  blob: Blob;
  mimeType: string;
  durationMs: number;
};

export class UnsupportedBrowserError extends Error {}

export class MicrophoneError extends Error {}

function readTrackSettings(stream: MediaStream): TrackSettings {
  const track = stream.getAudioTracks()[0];
  const settings: MediaTrackSettings | undefined = track?.getSettings();

  return {
    sampleRate: settings?.sampleRate ?? null,
    channelCount: settings?.channelCount ?? null,
    echoCancellation: settings?.echoCancellation ?? null,
    noiseSuppression: settings?.noiseSuppression ?? null,
    autoGainControl: settings?.autoGainControl ?? null,
  };
}

export function describeMicrophoneError(error: unknown): string {
  if (error instanceof UnsupportedBrowserError || error instanceof MicrophoneError) {
    return error.message;
  }
  if (error instanceof DOMException) {
    switch (error.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Microphone permission was denied. Allow microphone access in the browser and try again.";
      case "NotFoundError":
      case "DevicesNotFoundError":
        return "No microphone device was found.";
      case "NotReadableError":
      case "TrackStartError":
        return "The microphone is already in use by another application.";
      case "OverconstrainedError":
        return "The requested microphone settings are not supported by this device.";
      default:
        return `Microphone error: ${error.name}`;
    }
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "Unknown microphone error.";
}

/**
 * Thin wrapper around getUserMedia + MediaRecorder.
 *
 * - one microphone stream at a time, `start()` refuses while active
 * - every track is stopped on stop() and dispose(), so no stream leaks
 * - chunks are accumulated in memory (no streaming upload yet)
 */
export class MeetingRecorder {
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private finish: ((result: RecordingResult) => void) | null = null;
  private fail: ((error: Error) => void) | null = null;

  get isActive(): boolean {
    return this.stream !== null || this.recorder !== null;
  }

  /** Live microphone stream while recording; null otherwise. Read-only use. */
  get audioStream(): MediaStream | null {
    return this.stream;
  }

  async start(): Promise<RecorderSession> {
    if (this.isActive) {
      throw new MicrophoneError("A recording is already in progress.");
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      throw new UnsupportedBrowserError(
        "This browser does not support microphone capture (navigator.mediaDevices.getUserMedia is unavailable).",
      );
    }
    if (typeof window.MediaRecorder !== "function") {
      throw new UnsupportedBrowserError(
        "This browser does not support the MediaRecorder API.",
      );
    }

    const stream = await navigator.mediaDevices.getUserMedia(AUDIO_CONSTRAINTS);
    const requestedMimeType = getPreferredRecordingMimeType();

    const options: MediaRecorderOptions = {
      audioBitsPerSecond: REQUESTED_BITS_PER_SECOND,
    };
    if (requestedMimeType) {
      options.mimeType = requestedMimeType;
    }

    let recorder: MediaRecorder;
    try {
      recorder = new MediaRecorder(stream, options);
    } catch {
      try {
        recorder = new MediaRecorder(stream);
      } catch {
        stream.getTracks().forEach((track) => {
          track.stop();
        });
        throw new UnsupportedBrowserError(
          "MediaRecorder could not be started with this microphone stream.",
        );
      }
    }

    this.stream = stream;
    this.recorder = recorder;
    this.chunks = [];
    this.startedAt = Date.now();

    recorder.addEventListener("dataavailable", (event: BlobEvent) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    });
    recorder.addEventListener("error", () => {
      const failed = this.fail;
      this.finish = null;
      this.fail = null;
      this.cleanupStream();
      failed?.(new MicrophoneError("The browser reported a recording error."));
    });
    recorder.addEventListener("stop", () => {
      const finish = this.finish;
      this.finish = null;
      this.fail = null;
      this.cleanupStream();
      if (!finish) {
        return;
      }
      const mimeType = recorder.mimeType || this.chunks[0]?.type || "";
      const blob = new Blob(this.chunks, mimeType ? { type: mimeType } : undefined);
      finish({
        blob,
        mimeType,
        durationMs: Date.now() - this.startedAt,
      });
    });

    recorder.start(CHUNK_INTERVAL_MS);

    return {
      requestedMimeType,
      requestedBitsPerSecond: REQUESTED_BITS_PER_SECOND,
      actualBitsPerSecond: recorder.audioBitsPerSecond || null,
      trackSettings: readTrackSettings(stream),
    };
  }

  stop(): Promise<RecordingResult> {
    const recorder = this.recorder;
    if (!recorder || recorder.state === "inactive") {
      return Promise.reject(new MicrophoneError("No active recording to stop."));
    }

    return new Promise<RecordingResult>((resolve, reject) => {
      this.finish = resolve;
      this.fail = reject;
      recorder.stop();
    });
  }

  /** Stops all tracks and releases the recorder. Safe to call repeatedly. */
  dispose(): void {
    const cancelled = this.fail;
    this.finish = null;
    this.fail = null;
    if (this.recorder && this.recorder.state !== "inactive") {
      try {
        this.recorder.stop();
      } catch {
        /* already stopped */
      }
    }
    this.cleanupStream();
    this.recorder = null;
    this.chunks = [];
    cancelled?.(new MicrophoneError("Recording was cancelled."));
  }

  private cleanupStream(): void {
    this.stream?.getTracks().forEach((track) => {
      track.stop();
    });
    this.stream = null;
  }
}
