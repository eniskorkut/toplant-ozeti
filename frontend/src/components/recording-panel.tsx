"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AudioPlayer } from "@/components/audio-player";
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
import { ChevronDownIcon, MicIcon, StopIcon } from "@/lib/icons";

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

const PROVIDER_DESCRIPTIONS: Record<TranscriptionProviderId, string> = {
  local: "Ses cihazınızdan dışarı gönderilmez.",
  elevenlabs: "Daha hızlı ve daha yüksek transkripsiyon doğruluğu; ses ElevenLabs'a gönderilir.",
};

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

/** Lightweight live level meter: reads the active stream, never records it. */
function LevelMeter({ stream }: { stream: MediaStream | null }) {
  const barRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!stream || typeof window === "undefined") return;
    const AudioContextCtor =
      window.AudioContext ??
      (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextCtor) return;

    const context = new AudioContextCtor();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.75;
    source.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);

    let frame = 0;
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (let index = 0; index < data.length; index += 1) {
        const amplitude = Math.abs(data[index] - 128) / 128;
        if (amplitude > peak) peak = amplitude;
      }
      if (barRef.current) {
        barRef.current.style.transform = `scaleX(${Math.min(1, peak * 2.2)})`;
      }
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);

    return () => {
      window.cancelAnimationFrame(frame);
      source.disconnect();
      void context.close();
    };
  }, [stream]);

  if (!stream) return null;

  return (
    <div aria-hidden="true" className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-500/15">
      <span ref={barRef} className="block h-full origin-left scale-x-0 rounded-full bg-red-500/80" />
    </div>
  );
}

export function RecordingPanel() {
  const recorderRef = useRef<MeetingRecorder | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const startedAtRef = useRef(0);

  const [status, setStatus] = useState<Status>("idle");
  const [session, setSession] = useState<RecorderSession | null>(null);
  const [recording, setRecording] = useState<RecordingResult | null>(null);
  const [levelStream, setLevelStream] = useState<MediaStream | null>(null);
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
    setLevelStream(null);
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
      setLevelStream(recorder.audioStream ?? null);
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
      setLevelStream(null);
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
      setLevelStream(null);
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
      <div className="surface rounded-2xl p-4 sm:p-5">
        <h2
          id="recording"
          className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
        >
          Toplantı Kaydı
        </h2>

        {status === "recording" ? (
          <div className="mt-4 space-y-4">
            <div className="flex items-center justify-between gap-4">
              <span className="inline-flex items-center gap-2 text-sm font-medium text-red-600 dark:text-red-400">
                <span className="size-2 rounded-full bg-red-500" aria-hidden="true" />
                Kayıt sürüyor
              </span>
              <span className="font-mono text-3xl font-semibold tracking-tight tabular-nums text-zinc-900 dark:text-zinc-50">
                {formatDuration(elapsedMs / 1000)}
              </span>
            </div>

            <LevelMeter stream={levelStream} />

            <button
              type="button"
              onClick={handleStop}
              className="inline-flex items-center gap-2.5 rounded-2xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition-transform duration-160 ease-out hover:bg-zinc-800 active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              <StopIcon className="size-4" />
              Kaydı Durdur
            </button>
          </div>
        ) : (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={handleStart}
              disabled={busy}
              className="inline-flex items-center gap-3 rounded-2xl bg-zinc-900 py-2.5 pr-5 pl-2.5 text-sm font-medium text-white transition-transform duration-160 ease-out hover:bg-zinc-800 active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              <span className="grid size-9 place-items-center rounded-xl bg-white/10 dark:bg-zinc-900/10">
                <MicIcon className="size-5" />
              </span>
              {status === "requesting"
                ? "Mikrofon izni isteniyor…"
                : status === "uploading"
                  ? "Yükleniyor…"
                  : status === "processing"
                    ? "İşleniyor…"
                    : "Kaydı Başlat"}
            </button>
            {status === "idle" ? (
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                Kayıt bu cihazda tutulur; yüklemeyi siz başlatırsınız.
              </span>
            ) : null}
          </div>
        )}

        {status === "processing" ? (
          <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
            Ses yazıya dönüştürülüyor ve konuşmacılar ayrılıyor. Bu işlem kayıt süresine göre
            birkaç dakika sürebilir.
          </p>
        ) : null}

        {error ? (
          <p
            role="alert"
            className="mt-3 rounded-xl bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
          >
            {error}
          </p>
        ) : null}

        {processingError ? (
          <p
            role="alert"
            className="mt-3 rounded-xl bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
          >
            İşleme hatası: {processingError}
          </p>
        ) : null}

        {status === "uploaded" && meetingId ? (
          <div className="mt-5 space-y-5">
            <div className="rounded-xl bg-zinc-500/5 p-3">
              <div className="flex items-baseline justify-between gap-4">
                <span className="text-xs text-zinc-500 dark:text-zinc-400">Kayıt süresi</span>
                <span className="font-mono text-sm tabular-nums text-zinc-900 dark:text-zinc-100">
                  {formatDuration(uploadedDuration)}
                </span>
              </div>
              {recording && previewUrl ? (
                <div className="mt-2.5">
                  <AudioPlayer audioRef={previewAudioRef} src={previewUrl} label="Kayıt önizlemesi" />
                </div>
              ) : null}
            </div>

            <div className="space-y-2">
              <label
                htmlFor="speaker-count"
                className="block text-xs font-medium text-zinc-700 dark:text-zinc-300"
              >
                Konuşmacı sayısı
              </label>
              <select
                id="speaker-count"
                value={speakerChoice}
                onChange={(event) =>
                  setSpeakerChoice(event.target.value as (typeof SPEAKER_OPTIONS)[number])
                }
                className="w-full rounded-xl border border-zinc-950/10 bg-transparent px-3 py-2 text-sm text-zinc-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 sm:w-40 dark:border-white/15 dark:text-zinc-100"
              >
                <option value="auto">Otomatik</option>
                {SPEAKER_OPTIONS.filter((option) => option !== "auto").map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Konuşmacı sayısı yalnızca kümeleme içindir; kişiler anonim olarak
                etiketlenir (Kişi 1, Kişi 2, …).
              </p>
            </div>

            <fieldset className="space-y-2">
              <legend className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                Transkripsiyon yöntemi
              </legend>
              {providerOptions.map((option) => {
                const unavailable = !option.available;
                const selected = providerChoice === option.id;
                return (
                  <label
                    key={option.id}
                    className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-2.5 transition-colors duration-150 ease-out has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-zinc-500 ${
                      selected
                        ? "border-zinc-900 bg-zinc-500/5 dark:border-zinc-100"
                        : "border-zinc-950/10 hover:border-zinc-950/20 dark:border-white/15 dark:hover:border-white/25"
                    } ${unavailable ? "cursor-not-allowed opacity-60" : ""}`}
                  >
                    <input
                      type="radio"
                      name="transcription-provider"
                      value={option.id}
                      checked={selected}
                      disabled={unavailable}
                      onChange={() => setProviderChoice(option.id)}
                      className="mt-1 accent-zinc-900 dark:accent-zinc-100"
                    />
                    <span className="min-w-0">
                      <span className="block text-sm font-medium text-zinc-900 dark:text-zinc-100">
                        {option.id === "local" ? "Yerel" : option.label}
                      </span>
                      <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                        {PROVIDER_DESCRIPTIONS[option.id]}
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
                <label className="flex items-start gap-3 rounded-xl bg-amber-500/10 px-3 py-2.5 text-xs text-amber-900 dark:text-amber-200">
                  <input
                    type="checkbox"
                    checked={cloudAcknowledged}
                    onChange={(event) => setCloudAcknowledged(event.target.checked)}
                    className="mt-0.5 accent-amber-600"
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
                <div className="rounded-xl bg-zinc-500/5 px-3 py-2 text-xs text-zinc-700 dark:text-zinc-300">
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

            <button
              type="button"
              onClick={handleProcess}
              disabled={providerChoice === "elevenlabs" && !cloudAcknowledged}
              className="inline-flex w-full items-center justify-center rounded-2xl bg-zinc-900 px-5 py-3 text-sm font-medium text-white transition-transform duration-160 ease-out hover:bg-zinc-800 active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
            >
              Transkripsiyonu Başlat
            </button>

            <details className="group border-t border-zinc-950/5 pt-3 dark:border-white/10">
              <summary className="flex cursor-pointer list-none items-center gap-1.5 text-xs text-zinc-500 transition-colors duration-150 ease-out hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100">
                <ChevronDownIcon className="size-3.5 transition-transform duration-150 ease-out group-open:rotate-180" />
                Teknik ayrıntılar
              </summary>
              <dl className="mt-2 divide-y divide-zinc-950/5 dark:divide-white/5">
                {session ? (
                  <>
                    <MetadataRow
                      label="Kayıt formatı"
                      value={session.requestedMimeType ?? "tarayıcı varsayılanı"}
                    />
                    <MetadataRow
                      label="İstenen bit hızı"
                      value={formatBitrate(session.requestedBitsPerSecond)}
                    />
                    <MetadataRow
                      label="Gerçek bit hızı"
                      value={formatBitrate(session.actualBitsPerSecond)}
                    />
                  </>
                ) : null}
                {uploadedBytes != null ? (
                  <MetadataRow label="Yüklenen boyut" value={formatBytes(uploadedBytes)} />
                ) : null}
                <MetadataRow label="Toplantı kimliği" value={meetingId} />
              </dl>
            </details>
          </div>
        ) : null}

        {status === "completed" && meetingId ? (
          <div className="mt-4 rounded-xl bg-emerald-500/10 px-3 py-3">
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
