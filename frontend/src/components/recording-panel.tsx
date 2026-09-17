"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { uploadRecording, type RecordingCreated } from "@/lib/api";
import {
  MeetingRecorder,
  describeMicrophoneError,
  type RecorderSession,
  type RecordingResult,
} from "@/lib/audio-recorder";

type Status = "idle" | "requesting" | "recording" | "uploading" | "done";

function formatDuration(totalMs: number): string {
  const totalSeconds = Math.floor(totalMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatBitrate(bitsPerSecond: number | null): string {
  if (!bitsPerSecond) {
    return "not reported by browser";
  }
  return `${(bitsPerSecond / 1000).toFixed(0)} kbps`;
}

function MetadataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-xs text-zinc-500 dark:text-zinc-400">{label}</dt>
      <dd className="truncate font-mono text-xs text-zinc-900 tabular-nums dark:text-zinc-100">
        {value}
      </dd>
    </div>
  );
}

export function RecordingPanel() {
  const recorderRef = useRef<MeetingRecorder | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [session, setSession] = useState<RecorderSession | null>(null);
  const [recording, setRecording] = useState<RecordingResult | null>(null);
  const [result, setResult] = useState<RecordingCreated | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const startedAtRef = useRef(0);

  const releasePreview = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewUrl(null);
  }, []);

  useEffect(() => {
    return () => {
      recorderRef.current?.dispose();
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (status !== "recording") {
      return;
    }
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAtRef.current);
    }, 250);
    return () => window.clearInterval(timer);
  }, [status]);

  const handleStart = useCallback(async () => {
    setError(null);
    setResult(null);
    setRecording(null);
    setSession(null);
    setElapsedMs(0);
    releasePreview();
    setStatus("requesting");

    const recorder = recorderRef.current ?? new MeetingRecorder();
    recorderRef.current = recorder;

    try {
      const startedSession = await recorder.start();
      startedAtRef.current = Date.now();
      setSession(startedSession);
      setStatus("recording");
    } catch (startError) {
      recorder.dispose();
      setError(describeMicrophoneError(startError));
      setStatus("idle");
    }
  }, [releasePreview]);

  const handleStop = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) {
      return;
    }
    setStatus("uploading");
    setError(null);

    try {
      const stopped = await recorder.stop();
      setRecording(stopped);
      setElapsedMs(stopped.durationMs);

      const objectUrl = URL.createObjectURL(stopped.blob);
      previewUrlRef.current = objectUrl;
      setPreviewUrl(objectUrl);

      const uploaded = await uploadRecording(stopped.blob, {
        mimeType: stopped.mimeType,
        durationSeconds: stopped.durationMs / 1000,
        filename: stopped.mimeType.includes("mp4") ? "recording.mp4" : "recording.webm",
      });
      setResult(uploaded);
      setStatus("done");
    } catch (stopError) {
      setError(describeMicrophoneError(stopError));
      setStatus("idle");
    }
  }, []);

  return (
    <section aria-labelledby="recording" className="enter enter-2">
      <div className="surface rounded-2xl p-4">
        <h2
          id="recording"
          className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
        >
          Meeting Recording
        </h2>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          {status === "recording" ? (
            <>
              <span className="inline-flex items-center gap-2 text-sm text-zinc-900 dark:text-zinc-100">
                <span className="size-2 rounded-full bg-red-500" aria-hidden="true" />
                Recording
              </span>
              <span className="font-mono text-sm text-zinc-600 tabular-nums dark:text-zinc-400">
                {formatDuration(elapsedMs)}
              </span>
              <button
                type="button"
                onClick={handleStop}
                className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] dark:bg-zinc-100 dark:text-zinc-900"
              >
                Stop Recording
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleStart}
              disabled={status === "requesting" || status === "uploading"}
              className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
            >
              {status === "requesting"
                ? "Requesting microphone…"
                : status === "uploading"
                  ? "Uploading and converting…"
                  : "Start Recording"}
            </button>
          )}
        </div>

        {status === "uploading" ? (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
            Stopping the recorder, uploading the audio and converting it with ffmpeg.
          </p>
        ) : null}

        {error ? (
          <p
            role="alert"
            className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
          >
            {error}
          </p>
        ) : null}

        {session ? (
          <dl className="mt-3 divide-y divide-zinc-950/5 border-t border-zinc-950/5 dark:divide-white/5 dark:border-white/10">
            <MetadataRow
              label="Input MIME type"
              value={session.requestedMimeType ?? "browser default"}
            />
            <MetadataRow
              label="Requested bitrate"
              value={formatBitrate(session.requestedBitsPerSecond)}
            />
            <MetadataRow
              label="Actual MediaRecorder bitrate"
              value={formatBitrate(session.actualBitsPerSecond)}
            />
            <MetadataRow
              label="Microphone sample rate"
              value={
                session.trackSettings.sampleRate ? `${session.trackSettings.sampleRate} Hz` : "—"
              }
            />
            <MetadataRow
              label="Microphone channels"
              value={
                session.trackSettings.channelCount ? String(session.trackSettings.channelCount) : "—"
              }
            />
          </dl>
        ) : null}

        {recording ? (
          <div className="mt-3">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Recorded audio (local preview)</p>
            {previewUrl ? (
              <audio className="mt-1 w-full" controls src={previewUrl} />
            ) : null}
          </div>
        ) : null}

        {result ? (
          <div className="mt-4">
            <p className="text-sm text-zinc-900 dark:text-zinc-100">Recording complete</p>
            <dl className="mt-2 divide-y divide-zinc-950/5 border-t border-zinc-950/5 dark:divide-white/5 dark:border-white/10">
              <MetadataRow label="Duration" value={`${result.duration_seconds.toFixed(1)} s`} />
              <MetadataRow label="Input MIME type" value={result.input.mime_type} />
              <MetadataRow
                label="Requested bitrate"
                value={formatBitrate(session?.requestedBitsPerSecond ?? null)}
              />
              <MetadataRow
                label="Actual MediaRecorder bitrate"
                value={formatBitrate(session?.actualBitsPerSecond ?? null)}
              />
              <MetadataRow
                label="Original upload size"
                value={formatBytes(result.input.size_bytes)}
              />
              <MetadataRow label="MP3 size" value={formatBytes(result.mp3.size_bytes)} />
              <MetadataRow
                label="Processing WAV size"
                value={`${formatBytes(result.processing_wav.size_bytes)} (${result.processing_wav.sample_rate} Hz, ${result.processing_wav.channels} channel)`}
              />
              <MetadataRow
                label="Conversion duration"
                value={`${(result.conversion_ms / 1000).toFixed(3)} s (${result.conversion_ms} ms)`}
              />
              <MetadataRow label="Recording ID" value={result.recording_id} />
            </dl>
          </div>
        ) : null}
      </div>
    </section>
  );
}
