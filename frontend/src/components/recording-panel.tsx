"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  getElevenLabsUsage,
  getMeeting,
  getTranscriptionProviders,
  processMeeting,
  uploadRecording,
  type ElevenLabsUsage,
  type ProviderCapabilities,
  type TranscriptionProviderId,
} from "@/lib/api";
import {
  MeetingRecorder,
  describeMicrophoneError,
  type RecorderSession,
  type RecordingResult,
} from "@/lib/audio-recorder";
import { formatDuration, statusLabel } from "@/lib/format";

type Status =
  | "idle"
  | "requesting"
  | "recording"
  | "uploading"
  | "uploaded"
  | "processing"
  | "completed"
  | "failed";

const POLL_INTERVAL_MS = 2000;
const SPEAKER_OPTIONS = [
  "auto",
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8",
  "9",
  "10",
  "11",
  "12",
] as const;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatBitrate(bitsPerSecond: number | null): string {
  if (!bitsPerSecond) return "tarayıcı bildirmedi";
  return `${(bitsPerSecond / 1000).toFixed(0)} kbps`;
}

function UsageItem({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <dt className="inline text-zinc-500 dark:text-zinc-400">{label}: </dt>
      <dd className="inline font-mono tabular-nums">{value}</dd>
    </span>
  );
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
  const startedAtRef = useRef(0);

  const [status, setStatus] = useState<Status>("idle");
  const [session, setSession] = useState<RecorderSession | null>(null);
  const [recording, setRecording] = useState<RecordingResult | null>(null);
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [uploadedBytes, setUploadedBytes] = useState<number | null>(null);
  const [uploadedDuration, setUploadedDuration] = useState<number | null>(null);
  const [speakerChoice, setSpeakerChoice] = useState<(typeof SPEAKER_OPTIONS)[number]>("auto");
  const [providers, setProviders] = useState<ProviderCapabilities | null>(null);
  const [providerChoice, setProviderChoice] = useState<TranscriptionProviderId>("local");
  const [cloudAcknowledged, setCloudAcknowledged] = useState(false);
  const [usage, setUsage] = useState<ElevenLabsUsage | null>(null);
  const [usageError, setUsageError] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [processingError, setProcessingError] = useState<string | null>(null);

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
    let cancelled = false;
    getTranscriptionProviders()
      .then((capabilities) => {
        if (cancelled) return;
        setProviders(capabilities);
        setProviderChoice(capabilities.default);
      })
      .catch(() => {
        // Keep the local default when capabilities cannot be loaded.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Usage is fetched lazily when the cloud provider is selected; a failure never
  // blocks provider selection or transcription.
  useEffect(() => {
    if (providerChoice !== "elevenlabs" || usage || usageError) return;
    let cancelled = false;
    getElevenLabsUsage()
      .then((result) => {
        if (!cancelled) setUsage(result);
      })
      .catch(() => {
        if (!cancelled) setUsageError(true);
      });
    return () => {
      cancelled = true;
    };
  }, [providerChoice, usage, usageError]);

  useEffect(() => {
    if (status !== "recording") return;
    const timer = window.setInterval(() => {
      setElapsedMs(Date.now() - startedAtRef.current);
    }, 250);
    return () => window.clearInterval(timer);
  }, [status]);

  // Explicit polling loop: exactly one outstanding timer/request at a time.
  // The next poll is scheduled from inside the loop, so it never depends on a
  // state update to the same value ("processing" -> "processing") to continue.
  useEffect(() => {
    if (status !== "processing" || !meetingId) return;
    let cancelled = false;
    let timer: number | undefined;

    const poll = async () => {
      try {
        const meeting = await getMeeting(meetingId);
        if (cancelled) return;
        if (meeting.status === "completed") {
          setStatus("completed");
          return;
        }
        if (meeting.status === "failed") {
          setProcessingError(meeting.processing_error ?? "İşleme başarısız oldu.");
          setStatus("failed");
          return;
        }
        // Still queued/processing: keep polling.
        timer = window.setTimeout(poll, POLL_INTERVAL_MS);
      } catch (pollError) {
        if (cancelled) return;
        setProcessingError(
          pollError instanceof Error ? pollError.message : "Durum alınamadı.",
        );
        setStatus("failed");
      }
    };

    timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [status, meetingId]);

  const handleStart = useCallback(async () => {
    setError(null);
    setProcessingError(null);
    setRecording(null);
    setSession(null);
    setMeetingId(null);
    setUploadedBytes(null);
    setUploadedDuration(null);
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
    if (!recorder) return;
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
      setMeetingId(uploaded.meeting_id);
      setUploadedBytes(uploaded.input.size_bytes);
      setUploadedDuration(uploaded.duration_seconds);
      setStatus("uploaded");
    } catch (stopError) {
      setError(
        stopError instanceof ApiError || stopError instanceof Error
          ? stopError.message
          : "Yükleme başarısız oldu.",
      );
      setStatus("idle");
    }
  }, []);

  const handleProcess = useCallback(async () => {
    if (!meetingId) return;
    setError(null);
    setProcessingError(null);
    try {
      await processMeeting(meetingId, {
        speakerCount: speakerChoice === "auto" ? null : Number(speakerChoice),
        transcriptionProvider: providerChoice,
      });
      setStatus("processing");
    } catch (processError) {
      setError(
        processError instanceof Error
          ? processError.message
          : "İşleme kuyruğa alınamadı.",
      );
      setStatus("uploaded");
    }
  }, [meetingId, speakerChoice, providerChoice]);

  const busy = status === "requesting" || status === "uploading" || status === "processing";
  const providerOptions =
    providers?.providers ?? [
      { id: "local" as const, available: true, cloud: false, label: "Yerel" },
      {
        id: "elevenlabs" as const,
        available: false,
        cloud: true,
        label: "ElevenLabs",
      },
    ];

  return (
    <section aria-labelledby="recording" className="enter enter-2">
      <div className="surface rounded-2xl p-4">
        <h2
          id="recording"
          className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
        >
          Toplantı Kaydı
        </h2>

        <div className="mt-3 flex flex-wrap items-center gap-3">
          {status === "recording" ? (
            <>
              <span className="inline-flex items-center gap-2 text-sm text-zinc-900 dark:text-zinc-100">
                <span className="size-2 rounded-full bg-red-500" aria-hidden="true" />
                Kayıt sürüyor
              </span>
              <span className="font-mono text-sm text-zinc-600 tabular-nums dark:text-zinc-400">
                {formatDuration(elapsedMs / 1000)}
              </span>
              <button
                type="button"
                onClick={handleStop}
                className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] dark:bg-zinc-100 dark:text-zinc-900"
              >
                Kaydı Durdur
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={handleStart}
              disabled={busy}
              className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
            >
              {status === "requesting"
                ? "Mikrofon izni isteniyor…"
                : status === "uploading"
                  ? "Yükleniyor…"
                  : status === "processing"
                    ? "İşleniyor…"
                    : "Kaydı Başlat"}
            </button>
          )}
        </div>

        {status === "processing" ? (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
            Ses yazıya dönüştürülüyor ve konuşmacılar ayrılıyor. Bu işlem kayıt süresine göre
            birkaç dakika sürebilir.
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

        {processingError ? (
          <p
            role="alert"
            className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
          >
            İşleme hatası: {processingError}
          </p>
        ) : null}

        {session ? (
          <dl className="mt-3 divide-y divide-zinc-950/5 border-t border-zinc-950/5 dark:divide-white/5 dark:border-white/10">
            <MetadataRow label="Kayıt formatı" value={session.requestedMimeType ?? "tarayıcı varsayılanı"} />
            <MetadataRow label="İstenen bit hızı" value={formatBitrate(session.requestedBitsPerSecond)} />
            <MetadataRow label="Gerçek bit hızı" value={formatBitrate(session.actualBitsPerSecond)} />
          </dl>
        ) : null}

        {recording && previewUrl ? (
          <div className="mt-3">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">Kayıt önizlemesi</p>
            <audio className="mt-1 w-full" controls src={previewUrl} />
          </div>
        ) : null}

        {status === "uploaded" && meetingId ? (
          <div className="mt-4 space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <label
                htmlFor="speaker-count"
                className="text-xs text-zinc-500 dark:text-zinc-400"
              >
                Konuşmacı sayısı
              </label>
              <select
                id="speaker-count"
                value={speakerChoice}
                onChange={(event) =>
                  setSpeakerChoice(event.target.value as (typeof SPEAKER_OPTIONS)[number])
                }
                className="rounded-lg border border-zinc-950/10 bg-transparent px-2 py-1.5 text-sm text-zinc-900 dark:border-white/15 dark:text-zinc-100"
              >
                <option value="auto">Otomatik</option>
                {SPEAKER_OPTIONS.filter((option) => option !== "auto").map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleProcess}
                disabled={providerChoice === "elevenlabs" && !cloudAcknowledged}
                className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
              >
                Transkripsiyonu Başlat
              </button>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Konuşmacı sayısı yalnızca kümeleme içindir; kişiler anonim olarak
              etiketlenir (Kişi 1, Kişi 2, …).
            </p>

            <fieldset className="space-y-2">
              <legend className="text-xs text-zinc-500 dark:text-zinc-400">
                Transkripsiyon yöntemi
              </legend>
              {providerOptions.map((option) => {
                const unavailable = !option.available;
                const selected = providerChoice === option.id;
                return (
                  <label
                    key={option.id}
                    className={`flex cursor-pointer items-start gap-3 rounded-lg border px-3 py-2 ${
                      selected
                        ? "border-zinc-900 dark:border-zinc-100"
                        : "border-zinc-950/10 dark:border-white/15"
                    } ${unavailable ? "cursor-not-allowed opacity-60" : ""}`}
                  >
                    <input
                      type="radio"
                      name="transcription-provider"
                      value={option.id}
                      checked={selected}
                      disabled={unavailable}
                      onChange={() => setProviderChoice(option.id)}
                      className="mt-1"
                    />
                    <span className="min-w-0">
                      <span className="block text-sm text-zinc-900 dark:text-zinc-100">
                        {option.id === "local" ? "Yerel" : option.label}
                      </span>
                      <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                        {option.id === "local"
                          ? "Ses kaydı yerel işlem hattında işlenir."
                          : "Ses kaydı transkripsiyon için ElevenLabs hizmetine gönderilir."}
                      </span>
                      {unavailable ? (
                        <span className="mt-1 block text-xs text-amber-700 dark:text-amber-400">
                          ElevenLabs API yapılandırılmamış.
                        </span>
                      ) : null}
                    </span>
                  </label>
                );
              })}
            </fieldset>

            {providerChoice === "elevenlabs" ? (
              <>
                <label className="flex items-start gap-3 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-200">
                  <input
                    type="checkbox"
                    checked={cloudAcknowledged}
                    onChange={(event) => setCloudAcknowledged(event.target.checked)}
                    className="mt-0.5"
                  />
                  <span>
                    Ses kaydının transkripsiyon amacıyla ElevenLabs&apos;a gönderileceğini
                    anlıyorum.
                  </span>
                </label>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Konuşmacı sayısı ElevenLabs için beklenen azami sayıdır; sonuçta daha az
                  konuşmacı tespit edilebilir.
                </p>
                <div className="rounded-lg bg-zinc-500/10 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300">
                  {usage?.available ? (
                    <dl className="flex flex-wrap gap-x-4 gap-y-1">
                      {usage.tier ? <UsageItem label="Plan" value={usage.tier} /> : null}
                      {usage.usage != null ? (
                        <UsageItem label="Kullanım" value={String(usage.usage)} />
                      ) : null}
                      {usage.limit != null ? (
                        <UsageItem label="Limit" value={String(usage.limit)} />
                      ) : null}
                      {usage.remaining != null ? (
                        <UsageItem label="Kalan" value={String(usage.remaining)} />
                      ) : null}
                      {usage.reset_at ? (
                        <UsageItem
                          label="Yenilenme"
                          value={new Date(usage.reset_at).toLocaleDateString("tr-TR")}
                        />
                      ) : null}
                    </dl>
                  ) : (
                    <p>
                      Kullanım bilgisi ElevenLabs panelindeki Developers → Analytics → Usage
                      bölümünden görülebilir.
                    </p>
                  )}
                </div>
              </>
            ) : null}
            <dl className="divide-y divide-zinc-950/5 border-t border-zinc-950/5 dark:divide-white/5 dark:border-white/10">
              {uploadedDuration != null ? (
                <MetadataRow label="Kayıt süresi" value={formatDuration(uploadedDuration)} />
              ) : null}
              {uploadedBytes != null ? (
                <MetadataRow label="Yüklenen boyut" value={formatBytes(uploadedBytes)} />
              ) : null}
              <MetadataRow label="Kayıt kimliği" value={meetingId} />
            </dl>
          </div>
        ) : null}

        {status === "completed" && meetingId ? (
          <div className="mt-4 rounded-lg bg-emerald-500/10 px-3 py-3">
            <p className="text-sm text-emerald-800 dark:text-emerald-300">
              Transkripsiyon tamamlandı.
            </p>
            <Link
              href={`/meetings/${meetingId}`}
              className="mt-2 inline-block text-sm font-medium text-emerald-800 underline underline-offset-4 dark:text-emerald-300"
            >
              Toplantıyı aç
            </Link>
          </div>
        ) : null}

        {status === "failed" && meetingId ? (
          <div className="mt-4 space-y-2">
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Kayıt:{" "}
              <Link href={`/meetings/${meetingId}`} className="underline underline-offset-4">
                {meetingId}
              </Link>
            </p>
          </div>
        ) : null}

        {status === "processing" || status === "completed" ? (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
            Durum: {statusLabel(status)}
          </p>
        ) : null}
      </div>
    </section>
  );
}
