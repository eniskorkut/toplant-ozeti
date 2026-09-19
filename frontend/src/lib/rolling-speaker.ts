/**
 * Rolling speaker tracker: buffers realtime PCM chunks and posts overlapping
 * windows to our backend (which calls Scribe v2 batch for diarization).
 *
 * Windows never grow: only the newest `windowSeconds` are uploaded, so a long
 * meeting does not re-upload its whole history. Failures are non-fatal — live
 * text and the recording continue, speaker labels simply stay provisional.
 */

import type { SpeakerWindowResult } from "@/lib/api";

// Selected by measured benchmark (data/meetings/b1095740…): 6 s windows with 2 s
// overlap gave zero label switches/fragments on rapidly alternating speech, while
// 4 s windows fragmented; 8–10 s windows only added label delay.
export const WINDOW_SECONDS = 6;
export const OVERLAP_SECONDS = 2;
export const STEP_SECONDS = WINDOW_SECONDS - OVERLAP_SECONDS;
/** One delayed retry per window; after that the window is skipped, never looped. */
export const MAX_WINDOW_RETRIES = 2;
export const RETRY_DELAY_MS = 1500;

export type RollingTrackerOptions = {
  liveSessionId: string;
  speakerCount: number | null;
  send: (request: {
    pcm: Blob;
    startSeconds: number;
    endSeconds: number;
    sequence: number;
    speakerCount: number | null;
  }) => Promise<SpeakerWindowResult>;
  onResult: (result: SpeakerWindowResult) => void;
  onFailure?: (error: unknown) => void;
};

type Chunk = { pcm: Int16Array; startSeconds: number };

export class RollingSpeakerTracker {
  private chunks: Chunk[] = [];
  private nextWindowStart = 0;
  private sequence = 1;
  private inFlight = false;
  private stopped = false;
  private failureCount = 0;
  private retryTimer: number | null = null;
  private readonly step = STEP_SECONDS;

  constructor(private readonly options: RollingTrackerOptions) {}

  get uploadedSeconds(): number {
    return this.sequence === 1 ? 0 : (this.sequence - 1) * WINDOW_SECONDS;
  }

  get requestCount(): number {
    return this.sequence - 1;
  }

  /** Ingest one PCM chunk (16 kHz mono s16le) from the shared microphone path. */
  push(pcm: Int16Array, startSeconds: number): void {
    if (this.stopped) return;
    this.chunks.push({ pcm, startSeconds });
    this.maybeSend();
  }

  private maybeSend(): void {
    if (this.inFlight || this.stopped || this.retryTimer !== null) return;
    if (this.chunks.length === 0) return;
    const bufferEnd = this.endSeconds();
    if (bufferEnd - this.nextWindowStart < WINDOW_SECONDS) return;

    const start = this.nextWindowStart;
    const end = start + WINDOW_SECONDS;
    const pcm = this.slice(start, end);
    if (!pcm) return;

    const pcmBlob = new Blob([pcm.buffer as ArrayBuffer], {
      type: "application/octet-stream",
    });
    const sequence = this.sequence;
    this.inFlight = true;
    this.options
      .send({
        pcm: pcmBlob,
        startSeconds: start,
        endSeconds: end,
        sequence,
        speakerCount: this.options.speakerCount,
      })
      .then((result) => {
        this.failureCount = 0;
        this.sequence += 1;
        this.nextWindowStart += this.step;
        this.dropBefore(this.nextWindowStart);
        this.options.onResult(result);
      })
      .catch((error) => {
        this.failureCount += 1;
        this.options.onFailure?.(error);
        if (this.failureCount >= MAX_WINDOW_RETRIES) {
          // Give up on this window (bounded), keep the timeline consistent.
          this.failureCount = 0;
          this.sequence += 1;
          this.nextWindowStart += this.step;
          this.dropBefore(this.nextWindowStart);
        } else {
          this.retryTimer = window.setTimeout(() => {
            this.retryTimer = null;
            this.maybeSend();
          }, RETRY_DELAY_MS);
        }
      })
      .finally(() => {
        this.inFlight = false;
        this.maybeSend();
      });
  }

  private endSeconds(): number {
    const last = this.chunks[this.chunks.length - 1];
    return last.startSeconds + last.pcm.length / 16_000;
  }

  private slice(startSeconds: number, endSeconds: number): Int16Array | null {
    const startSample = Math.round(startSeconds * 16_000);
    const endSample = Math.round(endSeconds * 16_000);
    const output = new Int16Array(endSample - startSample);
    let written = 0;
    for (const chunk of this.chunks) {
      const chunkStart = Math.round(chunk.startSeconds * 16_000);
      const chunkEnd = chunkStart + chunk.pcm.length;
      const from = Math.max(chunkStart, startSample);
      const to = Math.min(chunkEnd, endSample);
      if (to <= from) continue;
      output.set(
        chunk.pcm.subarray(from - chunkStart, to - chunkStart),
        from - startSample,
      );
      written += to - from;
    }
    if (written === 0) return null;
    // The window must be complete: a partial tail would misalign timestamps.
    if (written < output.length * 0.5) return null;
    return output;
  }

  private dropBefore(startSeconds: number): void {
    const keep = Math.round(startSeconds * 16_000);
    this.chunks = this.chunks.filter(
      (chunk) => Math.round(chunk.startSeconds * 16_000) + chunk.pcm.length > keep,
    );
  }

  stop(): void {
    this.stopped = true;
    if (this.retryTimer !== null) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.chunks = [];
  }
}
