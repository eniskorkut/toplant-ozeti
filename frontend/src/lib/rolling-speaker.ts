/**
 * Long-context delayed speaker tracking: buffers realtime PCM and posts sliding
 * snapshots to our backend (ElevenLabs Scribe v2 batch diarization).
 *
 * Each snapshot re-reads a LONGER conversational context than the previous
 * architecture (lookback seconds instead of one short independent window), which
 * is what the full-file reference showed Scribe needs. Snapshots overlap heavily,
 * so the mapper has strong continuity evidence; memory stays bounded because old
 * chunks are dropped once they fall out of the lookback.
 *
 * Failures are non-fatal — live text and the recording continue, speaker labels
 * simply stay provisional.
 */

import type { SpeakerWindowResult } from "@/lib/api";

/** Selected by benchmark against the full-file reference (see diag harness). */
export const LOOKBACK_SECONDS = 12;
export const STEP_SECONDS = 4;
export const RETRY_DELAY_MS = 1500;
export const MAX_WINDOW_RETRIES = 2;
const FIRST_ATTEMPT_MULTIPLIER = 2;

export type RollingTrackerOptions = {
  liveSessionId: string;
  speakerCount: number | null;
  send: (request: {
    pcm: Blob;
    startSeconds: number;
    endSeconds: number;
    sequence: number;
    speakerCount: number | null;
    stableUntil: number;
  }) => Promise<SpeakerWindowResult>;
  onResult: (result: SpeakerWindowResult) => void;
  onFailure?: (error: unknown) => void;
};

type Chunk = { pcm: Int16Array; startSeconds: number };

export class RollingSpeakerTracker {
  private chunks: Chunk[] = [];
  private nextAttempt = STEP_SECONDS * FIRST_ATTEMPT_MULTIPLIER;
  private sequence = 1;
  private inFlight = false;
  private stopped = false;
  private failureCount = 0;
  private retryTimer: number | null = null;
  private uploadedSecondsTotal = 0;

  constructor(private readonly options: RollingTrackerOptions) {}

  get uploadedSeconds(): number {
    return Math.round(this.uploadedSecondsTotal * 1000) / 1000;
  }

  get requestCount(): number {
    return this.sequence - 1;
  }

  /** Ingest one PCM chunk (16 kHz mono s16le) from the shared microphone path. */
  push(pcm: Int16Array, startSeconds: number): void {
    if (this.stopped) return;
    this.chunks.push({ pcm, startSeconds });
    this.trim();
    this.maybeSend();
  }

  private endSeconds(): number {
    const last = this.chunks[this.chunks.length - 1];
    return last ? last.startSeconds + last.pcm.length / 16_000 : 0;
  }

  private trim(): void {
    const keepFrom = Math.max(0, this.nextAttempt - LOOKBACK_SECONDS);
    const keep = Math.round(keepFrom * 16_000);
    this.chunks = this.chunks.filter(
      (chunk) => Math.round(chunk.startSeconds * 16_000) + chunk.pcm.length > keep,
    );
  }

  private maybeSend(): void {
    if (this.inFlight || this.stopped || this.retryTimer !== null) return;
    if (this.chunks.length === 0) return;
    if (this.endSeconds() < this.nextAttempt) return;

    const snapshotEnd = this.nextAttempt;
    const snapshotStart = Math.max(0, snapshotEnd - LOOKBACK_SECONDS);
    const pcm = this.slice(snapshotStart, snapshotEnd);
    if (!pcm) return;

    const stableUntil = Math.max(snapshotStart, snapshotEnd - STEP_SECONDS);
    const sequence = this.sequence;
    const uploaded = snapshotEnd - snapshotStart;
    this.inFlight = true;
    this.options
      .send({
        pcm: new Blob([pcm.buffer as ArrayBuffer], { type: "application/octet-stream" }),
        startSeconds: snapshotStart,
        endSeconds: snapshotEnd,
        sequence,
        speakerCount: this.options.speakerCount,
        stableUntil,
      })
      .then((result) => {
        // stop() is final: a late response from a previous recording must never
        // mutate state, advance the sequence or label the next recording.
        if (this.stopped) return;
        this.failureCount = 0;
        this.sequence += 1;
        this.uploadedSecondsTotal += uploaded;
        this.nextAttempt += STEP_SECONDS;
        this.options.onResult(result);
      })
      .catch((error) => {
        if (this.stopped) return;
        this.failureCount += 1;
        this.options.onFailure?.(error);
        if (this.failureCount >= MAX_WINDOW_RETRIES) {
          // Bounded: skip this snapshot and continue with the next one.
          this.failureCount = 0;
          this.sequence += 1;
          this.nextAttempt += STEP_SECONDS;
        } else {
          this.retryTimer = window.setTimeout(() => {
            this.retryTimer = null;
            this.maybeSend();
          }, RETRY_DELAY_MS);
        }
      })
      .finally(() => {
        this.inFlight = false;
        if (this.stopped) return;
        this.maybeSend();
      });
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
      output.set(chunk.pcm.subarray(from - chunkStart, to - chunkStart), from - startSample);
      written += to - from;
    }
    if (written === 0) return null;
    // Require most of the snapshot: a sparse tail would misalign timestamps.
    if (written < output.length * 0.5) return null;
    return output;
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
