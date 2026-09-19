/**
 * 16 kHz mono PCM capture over an existing microphone stream.
 *
 * The stream is shared with MediaRecorder: this module only *reads* it through an
 * AudioWorklet, resamples to the Scribe realtime format (16-bit little-endian PCM
 * at 16 kHz) and emits small chunks (~100 ms) so live text stays low-latency.
 */

export const TARGET_SAMPLE_RATE = 16_000;
export const EMIT_SAMPLES = 1_600; // 100 ms at 16 kHz

/** Streaming linear resampler; keeps fractional position between frames. */
export class Resampler {
  private readonly ratio: number;
  private readonly buffer: number[] = [];
  private position = 0;

  constructor(
    private readonly inputRate: number,
    private readonly outputRate: number = TARGET_SAMPLE_RATE,
  ) {
    this.ratio = inputRate / outputRate;
  }

  /** Push a frame and return the resampled output (may be empty). */
  push(frame: Float32Array): Float32Array {
    for (let index = 0; index < frame.length; index += 1) {
      this.buffer.push(frame[index]);
    }

    const output: number[] = [];
    while (this.position + 1 < this.buffer.length) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      const value = this.buffer[left] * (1 - fraction) + this.buffer[left + 1] * fraction;
      output.push(value);
      this.position += this.ratio;
    }

    const consumed = Math.floor(this.position);
    if (consumed > 0) {
      this.buffer.splice(0, consumed);
      this.position -= consumed;
    }
    return Float32Array.from(output);
  }
}

export function floatToInt16(samples: Float32Array): Int16Array {
  const output = new Int16Array(samples.length);
  for (let index = 0; index < samples.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[index]));
    output[index] = clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);
  }
  return output;
}

const WORKLET_SOURCE = `
class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(input[0].slice(0));
    }
    return true;
  }
}
registerProcessor("pcm-capture", PcmCaptureProcessor);
`;

export type PcmCaptureOptions = {
  /** Receives 16 kHz mono s16le chunks; `startSeconds` is the audio-timeline offset. */
  onChunk: (pcm: ArrayBuffer, startSeconds: number) => void;
  onError?: (error: Error) => void;
};

export type PcmCaptureStats = {
  chunks: number;
  streamedSeconds: number;
  /** Wall-clock gaps between emitted chunks (dev instrumentation only). */
  averageIntervalMs: number | null;
  maxIntervalMs: number | null;
  gapCount: number;
};

const GAP_THRESHOLD_MS = 250;

export class PcmCapture {
  private context: AudioContext | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private node: AudioWorkletNode | null = null;
  private resampler: Resampler = new Resampler(48_000, TARGET_SAMPLE_RATE);
  private pending: number[] = [];
  private emittedSamples = 0;
  private emittedChunks = 0;
  private stopped = false;
  private lastEmitAtMs: number | null = null;
  private intervals: number[] = [];
  private gapCount = 0;

  constructor(
    private readonly stream: MediaStream,
    private readonly options: PcmCaptureOptions,
  ) {}

  get nativeSampleRate(): number | null {
    return this.context?.sampleRate ?? null;
  }

  get streamedSeconds(): number {
    return this.emittedSamples / TARGET_SAMPLE_RATE;
  }

  get stats(): PcmCaptureStats {
    const intervals = this.intervals;
    return {
      chunks: this.emittedChunks,
      streamedSeconds: this.emittedSamples / TARGET_SAMPLE_RATE,
      averageIntervalMs:
        intervals.length > 0
          ? Math.round((intervals.reduce((sum, value) => sum + value, 0) / intervals.length) * 10) /
            10
          : null,
      maxIntervalMs:
        intervals.length > 0 ? Math.round(Math.max(...intervals) * 10) / 10 : null,
      gapCount: this.gapCount,
    };
  }

  /**
   * Feed one AudioWorklet frame (native rate). Public so cadence can be verified
   * without a real AudioContext; the worklet handler calls this too.
   */
  handleFrame(frame: Float32Array): void {
    if (this.stopped) return;
    try {
      const resampled = this.resampler.push(frame);
      if (resampled.length === 0) return;
      for (let index = 0; index < resampled.length; index += 1) {
        this.pending.push(resampled[index]);
      }
      this.flushCompletedChunks();
    } catch (error) {
      this.options.onError?.(error instanceof Error ? error : new Error(String(error)));
    }
  }

  async start(): Promise<number> {
    const AudioContextCtor =
      window.AudioContext ??
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) {
      throw new Error("AudioContext is not available in this browser");
    }

    const context = new AudioContextCtor();
    this.context = context;
    await context.resume();

    const moduleUrl = URL.createObjectURL(
      new Blob([WORKLET_SOURCE], { type: "application/javascript" }),
    );
    try {
      await context.audioWorklet.addModule(moduleUrl);
    } finally {
      URL.revokeObjectURL(moduleUrl);
    }

    const source = context.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(context, "pcm-capture");
    this.source = source;
    this.node = node;
    this.resampler = new Resampler(context.sampleRate, TARGET_SAMPLE_RATE);

    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
      if (this.stopped) return;
      const frame = event.data instanceof Float32Array ? event.data : new Float32Array();
      this.handleFrame(frame);
    };

    source.connect(node);
    // The worklet must be connected to the graph to be pulled; a zero gain keeps
    // it silent so it never reaches the speakers and never affects MediaRecorder.
    const silence = context.createGain();
    silence.gain.value = 0;
    node.connect(silence).connect(context.destination);

    return context.sampleRate;
  }

  private flushCompletedChunks(): void {
    while (this.pending.length >= EMIT_SAMPLES) {
      const slice = Float32Array.from(this.pending.splice(0, EMIT_SAMPLES));
      const pcm = floatToInt16(slice);
      const startSeconds = this.emittedSamples / TARGET_SAMPLE_RATE;
      this.emittedSamples += pcm.length;
      this.emittedChunks += 1;
      const now = performance.now();
      if (this.lastEmitAtMs !== null) {
        const interval = now - this.lastEmitAtMs;
        this.intervals.push(interval);
        if (interval > GAP_THRESHOLD_MS) {
          this.gapCount += 1;
        }
      }
      this.lastEmitAtMs = now;
      this.options.onChunk(pcm.buffer as ArrayBuffer, startSeconds);
    }
  }

  stop(): void {
    this.stopped = true;
    if (this.node) {
      this.node.port.onmessage = null;
      this.node.disconnect();
    }
    this.source?.disconnect();
    if (this.context) {
      void this.context.close();
    }
    this.node = null;
    this.source = null;
    this.context = null;
    this.pending = [];
  }
}
