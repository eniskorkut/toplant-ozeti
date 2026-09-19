"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { AudioPlayer } from "@/components/audio-player";
import { LiveTranscript, type LiveLine } from "@/components/live-transcript";
import { SpeakerRenameDialog } from "@/components/speaker-rename-dialog";
import {
  ApiError,
  clearLiveSpeakerAlias,
  clearMeetingSpeakerAlias,
  createLiveSession,
  deleteLiveSession,
  getElevenLabsUsage,
  getMeeting,
  getRealtimeToken,
  getTranscriptionProviders,
  processMeeting,
  sendSpeakerWindow,
  setLiveSpeakerAlias,
  setMeetingSpeakerAlias,
  uploadRecording,
  type ElevenLabsUsage,
  type ProviderCapabilities,
  type SpeakerWindowResult,
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
import { PcmCapture } from "@/lib/pcm-capture";
import { RollingSpeakerTracker } from "@/lib/rolling-speaker";
import { ScribeRealtimeClient, type RealtimeStatus } from "@/lib/scribe-realtime";
import { applyAlias, type SpeakerAliases } from "@/lib/speakers";

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
  elevenlabs:
    "Daha hızlı ve daha yüksek transkripsiyon doğruluğu; canlı transkript ve konuşmacı etiketleri için ses ElevenLabs'a gönderilir.",
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

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
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

function deriveLiveStatus(
  lines: LiveLine[],
  realtimeStatus: RealtimeStatus,
): string {
  if (realtimeStatus === "connecting" || realtimeStatus === "reconnecting") {
    return "Bağlanıyor…";
  }
  if (realtimeStatus === "idle") return "Canlı transkript";
  if (lines.length === 0) return "Dinleniyor";
  if (lines.some((line) => line.provisional)) return "Metin işleniyor";
  if (lines.some((line) => !line.canonical)) return "Konuşmacılar eşleştiriliyor";
  return "Dinleniyor";
}

export function RecordingPanel() {
  const recorderRef = useRef<MeetingRecorder | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const startedAtRef = useRef(0);

  const scribeRef = useRef<ScribeRealtimeClient | null>(null);
  const pcmRef = useRef<PcmCapture | null>(null);
  const trackerRef = useRef<RollingSpeakerTracker | null>(null);
  const lastRollingResultRef = useRef<SpeakerWindowResult | null>(null);
  const captureStartedAtRef = useRef<number | null>(null);
  const firstPcmAtRef = useRef<number | null>(null);
  const wsConnectedAtRef = useRef<number | null>(null);
  const firstPartialEventAtRef = useRef<number | null>(null);
  const firstPartialRenderedAtRef = useRef<number | null>(null);
  const pcmStatsRef = useRef<{
    chunks: number;
    averageIntervalMs: number | null;
    maxIntervalMs: number | null;
    gapCount: number;
  } | null>(null);
  const lineIdRef = useRef(0);
  const lastAudioEndRef = useRef<number | null>(null);
  const partialSeenRef = useRef(false);
  const metricsRef = useRef<{ partial: number[]; committed: number[]; label: number[] }>({
    partial: [],
    committed: [],
    label: [],
  });

  const [status, setStatus] = useState<Status>("idle");
  const [session, setSession] = useState<RecorderSession | null>(null);
  const [recording, setRecording] = useState<RecordingResult | null>(null);
  const [levelStream, setLevelStream] = useState<MediaStream | null>(null);
  const [meetingId, setMeetingId] = useState<string | null>(null);
  const [uploadedBytes, setUploadedBytes] = useState<number | null>(null);
  const [uploadedDuration, setUploadedDuration] = useState<number | null>(null);
  const [speakerChoice, setSpeakerChoice] = useState<(typeof SPEAKER_OPTIONS)[number]>("auto");
  const [liveSpeakerChoice, setLiveSpeakerChoice] =
    useState<(typeof SPEAKER_OPTIONS)[number]>("auto");
  const [providers, setProviders] = useState<ProviderCapabilities | null>(null);
  const [providerChoice, setProviderChoice] = useState<TranscriptionProviderId>("local");
  const [cloudAcknowledged, setCloudAcknowledged] = useState(false);
  const [usage, setUsage] = useState<ElevenLabsUsage | null>(null);
  const [usageError, setUsageError] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [processingError, setProcessingError] = useState<string | null>(null);

  // Live (ElevenLabs only) state
  const [liveSessionId, setLiveSessionId] = useState<string | null>(null);
  const [liveLines, setLiveLines] = useState<LiveLine[]>([]);
  const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("idle");
  const [liveWarning, setLiveWarning] = useState<string | null>(null);
  const [aliases, setAliases] = useState<SpeakerAliases>({});
  const [renameSpeaker, setRenameSpeaker] = useState<string | null>(null);
  const [renameBusy, setRenameBusy] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<{
    partial: number | null;
    committed: number | null;
    label: number | null;
    rollingSeconds: number;
    requests: number;
    firstPcmMs: number | null;
    wsConnectedMs: number | null;
    firstPartialVisibleMs: number | null;
    partialEventToVisibleMs: number | null;
    pcmChunks: number | null;
    pcmAverageIntervalMs: number | null;
    pcmMaxIntervalMs: number | null;
    pcmGapCount: number | null;
  } | null>(null);

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
      scribeRef.current?.close();
      pcmRef.current?.stop();
      trackerRef.current?.stop();
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

  // --- live pipeline callbacks -------------------------------------------

  const elapsedSeconds = useCallback(() => (Date.now() - startedAtRef.current) / 1000, []);

  const attachSpeakerLabels = useCallback((result: SpeakerWindowResult) => {
    lastRollingResultRef.current = result;
    setLiveLines((previous) =>
      previous.map((line) => {
        if (line.canonical) return line;
        const midpoint = (line.startSeconds + line.endSeconds) / 2;
        const match = result.assignments.find(
          (assignment) => midpoint >= assignment.start - 0.5 && midpoint <= assignment.end + 0.5,
        );
        if (!match) return line;
        if (!line.provisional) {
          metricsRef.current.label.push(Math.max(0, elapsedSeconds() - line.endSeconds));
        }
        return { ...line, canonical: match.canonical_speaker };
      }),
    );
  }, [elapsedSeconds]);

  const markPartialRendered = useCallback(() => {
    if (firstPartialRenderedAtRef.current !== null) return;
    const schedule =
      typeof requestAnimationFrame === "function"
        ? requestAnimationFrame
        : (callback: FrameRequestCallback) =>
            window.setTimeout(() => callback(performance.now()), 16);
    schedule(() => {
      if (firstPartialRenderedAtRef.current === null) {
        firstPartialRenderedAtRef.current = performance.now();
      }
    });
  }, []);

  const handlePartial = useCallback(
    (text: string) => {
      markPartialRendered();
      const now = elapsedSeconds();
      if (!partialSeenRef.current && lastAudioEndRef.current !== null) {
        metricsRef.current.partial.push(Math.max(0, now - lastAudioEndRef.current));
      }
      partialSeenRef.current = true;
      setLiveLines((previous) => {
        const next = [...previous];
        const last = next[next.length - 1];
        if (last && last.provisional) {
          next[next.length - 1] = { ...last, text, endSeconds: now };
        } else {
          next.push({
            id: lineIdRef.current + 1,
            startSeconds: now,
            endSeconds: now,
            text,
            canonical: null,
            provisional: true,
          });
          lineIdRef.current += 1;
        }
        return next;
      });
    },
    [elapsedSeconds, markPartialRendered],
  );

  const handleCommitted = useCallback(
    (text: string, words: { text: string; start: number; end: number }[]) => {
      const now = elapsedSeconds();
      const start = words.length > 0 ? words[0].start : now;
      const end = words.length > 0 ? words[words.length - 1].end : now;
      metricsRef.current.committed.push(Math.max(0, now - end));
      lastAudioEndRef.current = end;
      partialSeenRef.current = false;
      setLiveLines((previous) => {
        const next = previous.filter((line) => !line.provisional);
        next.push({
          id: lineIdRef.current + 1,
          startSeconds: start,
          endSeconds: end,
          text,
          canonical: null,
          provisional: false,
        });
        lineIdRef.current += 1;
        return next;
      });
      if (lastRollingResultRef.current) {
        attachSpeakerLabels(lastRollingResultRef.current);
      }
    },
    [attachSpeakerLabels, elapsedSeconds],
  );

  const startLivePipeline = useCallback(
    async (recorder: MeetingRecorder) => {
      const stream = recorder.audioStream;
      if (!stream) {
        setLiveWarning("Canlı transkript başlatılamadı — kayıt devam ediyor.");
        return;
      }
      let createdSessionId: string | null = null;
      try {
        const session = await createLiveSession();
        createdSessionId = session.live_session_id;
        setLiveSessionId(createdSessionId);

        const tracker = new RollingSpeakerTracker({
          liveSessionId: createdSessionId,
          speakerCount:
            liveSpeakerChoice === "auto" ? null : Number(liveSpeakerChoice),
          send: (request) => sendSpeakerWindow(createdSessionId as string, request),
          onResult: (result) => attachSpeakerLabels(result),
          onFailure: () => {
            setLiveWarning(
              "Konuşmacı etiketleri şu an alınamıyor; kayıt ve canlı metin devam ediyor.",
            );
          },
        });
        trackerRef.current = tracker;

        const scribe = new ScribeRealtimeClient({
          onPartial: handlePartial,
          onCommitted: handleCommitted,
          onStatus: (next) => {
            setRealtimeStatus(next);
            if (next === "connected") {
              setLiveWarning(null);
            } else if (next === "reconnecting" || next === "failed") {
              setLiveWarning(
                "Canlı transkript bağlantısı kesildi — kayıt devam ediyor.",
              );
            }
          },
          onTimeline: (event, atMs) => {
            if (event === "connected" && wsConnectedAtRef.current === null) {
              wsConnectedAtRef.current = atMs;
            }
            if (event === "partial_event" && firstPartialEventAtRef.current === null) {
              firstPartialEventAtRef.current = atMs;
            }
          },
        });
        scribeRef.current = scribe;

        const capture = new PcmCapture(stream, {
          onChunk: (pcm, startSeconds) => {
            if (firstPcmAtRef.current === null) {
              firstPcmAtRef.current = performance.now();
            }
            scribe.sendAudio(pcm);
            trackerRef.current?.push(new Int16Array(pcm), startSeconds);
          },
          onError: () => {
            setLiveWarning("Canlı ses işleme hatası — kayıt devam ediyor.");
          },
        });
        pcmRef.current = capture;
        captureStartedAtRef.current = performance.now();
        await capture.start();
        // Every connection attempt (including reconnects) mints a fresh
        // single-use token; consumed tokens are never reused.
        scribe.open(async () => (await getRealtimeToken()).token);
      } catch {
        if (createdSessionId) {
          void deleteLiveSession(createdSessionId).catch(() => undefined);
        }
        setLiveSessionId(null);
        setLiveWarning("Canlı transkript başlatılamadı — kayıt devam ediyor.");
      }
    },
    [attachSpeakerLabels, handleCommitted, handlePartial, liveSpeakerChoice],
  );

  const stopLivePipeline = useCallback(() => {
    const tracker = trackerRef.current;
    const captureStats = pcmRef.current?.stats ?? null;
    if (captureStats) pcmStatsRef.current = captureStats;
    scribeRef.current?.commit();
    scribeRef.current?.close();
    scribeRef.current = null;
    pcmRef.current?.stop();
    pcmRef.current = null;
    tracker?.stop();
    if (tracker) {
      const captureAt = captureStartedAtRef.current;
      const relative = (value: number | null) =>
        captureAt !== null && value !== null ? Math.max(0, value - captureAt) : null;
      const stats = pcmStatsRef.current;
      setLiveMetrics({
        partial: median(metricsRef.current.partial),
        committed: median(metricsRef.current.committed),
        label: median(metricsRef.current.label),
        rollingSeconds: tracker.uploadedSeconds,
        requests: tracker.requestCount,
        firstPcmMs: relative(firstPcmAtRef.current),
        wsConnectedMs: relative(wsConnectedAtRef.current),
        firstPartialVisibleMs: relative(firstPartialRenderedAtRef.current),
        partialEventToVisibleMs:
          firstPartialEventAtRef.current !== null &&
          firstPartialRenderedAtRef.current !== null
            ? Math.max(
                0,
                firstPartialRenderedAtRef.current - firstPartialEventAtRef.current,
              )
            : null,
        pcmChunks: stats?.chunks ?? null,
        pcmAverageIntervalMs: stats?.averageIntervalMs ?? null,
        pcmMaxIntervalMs: stats?.maxIntervalMs ?? null,
        pcmGapCount: stats?.gapCount ?? null,
      });
    }
    setRealtimeStatus("disconnected");
  }, []);

  // --- recording lifecycle -------------------------------------------------

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
    setLiveLines([]);
    setLiveWarning(null);
    setLiveMetrics(null);
    setRealtimeStatus("idle");
    lineIdRef.current = 0;
    lastAudioEndRef.current = null;
    partialSeenRef.current = false;
    captureStartedAtRef.current = null;
    firstPcmAtRef.current = null;
    wsConnectedAtRef.current = null;
    firstPartialEventAtRef.current = null;
    firstPartialRenderedAtRef.current = null;
    pcmStatsRef.current = null;
    metricsRef.current = { partial: [], committed: [], label: [] };
    releasePreview();
    setStatus("requesting");

    const recorder = recorderRef.current ?? new MeetingRecorder();
    recorderRef.current = recorder;
    const wantsLive = providerChoice === "elevenlabs" && cloudAcknowledged;
    try {
      const startedSession = await recorder.start();
      startedAtRef.current = Date.now();
      setSession(startedSession);
      setLevelStream(recorder.audioStream ?? null);
      setStatus("recording");
      if (wantsLive) {
        // Live failures never stop the recording.
        await startLivePipeline(recorder);
      }
    } catch (startError) {
      recorder.dispose();
      setError(describeMicrophoneError(startError));
      setStatus("idle");
    }
  }, [cloudAcknowledged, providerChoice, releasePreview, startLivePipeline]);

  const handleStop = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    if (providerChoice === "elevenlabs" && cloudAcknowledged) {
      stopLivePipeline();
    }
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
        liveSessionId,
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
  }, [cloudAcknowledged, liveSessionId, providerChoice, stopLivePipeline]);

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

  // --- rename ---------------------------------------------------------------

  const handleRenameSave = useCallback(
    async (displayName: string) => {
      if (!renameSpeaker) return;
      const canonical = renameSpeaker;
      setRenameBusy(true);
      setRenameError(null);
      try {
        if (status === "recording" && liveSessionId) {
          await setLiveSpeakerAlias(liveSessionId, canonical, displayName);
        } else if (meetingId) {
          await setMeetingSpeakerAlias(meetingId, canonical, displayName);
        }
        setAliases((previous) => applyAlias(previous, canonical, displayName));
        setRenameSpeaker(null);
      } catch (renameFailure) {
        setRenameError(
          renameFailure instanceof Error ? renameFailure.message : "Ad kaydedilemedi.",
        );
      } finally {
        setRenameBusy(false);
      }
    },
    [liveSessionId, meetingId, renameSpeaker, status],
  );

  const handleRenameReset = useCallback(async () => {
    if (!renameSpeaker) return;
    const canonical = renameSpeaker;
    setRenameBusy(true);
    setRenameError(null);
    try {
      if (status === "recording" && liveSessionId) {
        await clearLiveSpeakerAlias(liveSessionId, canonical);
      } else if (meetingId) {
        await clearMeetingSpeakerAlias(meetingId, canonical);
      }
      setAliases((previous) => applyAlias(previous, canonical, null));
      setRenameSpeaker(null);
    } catch (renameFailure) {
      setRenameError(
        renameFailure instanceof Error ? renameFailure.message : "Ad sıfırlanamadı.",
      );
    } finally {
      setRenameBusy(false);
    }
  }, [liveSessionId, meetingId, renameSpeaker, status]);

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
  const liveMode = providerChoice === "elevenlabs" && cloudAcknowledged;
  const liveStatusText = liveWarning ?? deriveLiveStatus(liveLines, realtimeStatus);
  const canStart = !busy && !(providerChoice === "elevenlabs" && !cloudAcknowledged);

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

            {liveMode ? (
              <LiveTranscript
                lines={liveLines}
                aliases={aliases}
                status={liveStatusText}
                warning={liveWarning}
                onRenameSpeaker={(canonical) => {
                  setRenameError(null);
                  setRenameSpeaker(canonical);
                }}
              />
            ) : null}
          </div>
        ) : (
          <div className="mt-4 space-y-5">
            {status === "idle" || status === "requesting" ? (
              <>
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
                        Ses kaydının canlı transkript, konuşmacı ayrımı ve nihai transkripsiyon
                        için ElevenLabs&apos;a gönderileceğini anlıyorum.
                      </span>
                    </label>

                    <div className="space-y-2">
                      <label
                        htmlFor="live-speaker-count"
                        className="block text-xs font-medium text-zinc-700 dark:text-zinc-300"
                      >
                        Konuşmacı sayısı (canlı)
                      </label>
                      <select
                        id="live-speaker-count"
                        value={liveSpeakerChoice}
                        onChange={(event) =>
                          setLiveSpeakerChoice(
                            event.target.value as (typeof SPEAKER_OPTIONS)[number],
                          )
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
                        Bilinen sayı, ElevenLabs için beklenen azami sayıdır; sonuçta daha az
                        konuşmacı tespit edilebilir.
                      </p>
                    </div>

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
                              value={new Date(usage.reset_at).toLocaleDateString("tr-TR", {
                                timeZone: "Europe/Istanbul",
                              })}
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

                <div className="flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={handleStart}
                    disabled={!canStart}
                    className="inline-flex items-center gap-3 rounded-2xl bg-zinc-900 py-2.5 pr-5 pl-2.5 text-sm font-medium text-white transition-transform duration-160 ease-out hover:bg-zinc-800 active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
                  >
                    <span className="grid size-9 place-items-center rounded-xl bg-white/10 dark:bg-zinc-900/10">
                      <MicIcon className="size-5" />
                    </span>
                    {status === "requesting" ? "Mikrofon izni isteniyor…" : "Kaydı Başlat"}
                  </button>
                  {status === "idle" ? (
                    <span className="text-xs text-zinc-500 dark:text-zinc-400">
                      {providerChoice === "elevenlabs"
                        ? "Canlı transkript kayıt sırasında görünür."
                        : "Kayıt bu cihazda tutulur; yüklemeyi siz başlatırsınız."}
                    </span>
                  ) : null}
                </div>
              </>
            ) : null}

            {status === "uploading" ? (
              <button
                type="button"
                disabled
                className="inline-flex items-center gap-2.5 rounded-2xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white opacity-60 dark:bg-zinc-100 dark:text-zinc-900"
              >
                Yükleniyor…
              </button>
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

            <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
              <span>Kayıt sırasında seçilen yöntem:</span>
              <span className="rounded-full bg-zinc-500/10 px-2 py-0.5 font-medium text-zinc-700 dark:text-zinc-300">
                {providerChoice === "elevenlabs" ? "ElevenLabs" : "Yerel"}
              </span>
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

            <button
              type="button"
              onClick={handleProcess}
              className="inline-flex w-full items-center justify-center rounded-2xl bg-zinc-900 px-5 py-3 text-sm font-medium text-white transition-transform duration-160 ease-out hover:bg-zinc-800 active:scale-[0.96] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 sm:w-auto dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
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
                {liveMetrics ? (
                  <>
                    <MetadataRow
                      label="İlk PCM parçası (yakalamadan)"
                      value={
                        liveMetrics.firstPcmMs == null
                          ? "—"
                          : `${(liveMetrics.firstPcmMs / 1000).toFixed(2)} sn`
                      }
                    />
                    <MetadataRow
                      label="WebSocket bağlantısı (yakalamadan)"
                      value={
                        liveMetrics.wsConnectedMs == null
                          ? "—"
                          : `${(liveMetrics.wsConnectedMs / 1000).toFixed(2)} sn`
                      }
                    />
                    <MetadataRow
                      label="İlk kısmi metin görünür (yakalamadan)"
                      value={
                        liveMetrics.firstPartialVisibleMs == null
                          ? "—"
                          : `${(liveMetrics.firstPartialVisibleMs / 1000).toFixed(2)} sn`
                      }
                    />
                    <MetadataRow
                      label="Kısmi olay → görünür"
                      value={
                        liveMetrics.partialEventToVisibleMs == null
                          ? "—"
                          : `${liveMetrics.partialEventToVisibleMs.toFixed(0)} ms`
                      }
                    />
                    <MetadataRow
                      label="PCM parçaları"
                      value={
                        liveMetrics.pcmChunks == null
                          ? "—"
                          : `${liveMetrics.pcmChunks} parça · ort. ${
                              liveMetrics.pcmAverageIntervalMs?.toFixed(0) ?? "—"
                            } ms · en uzun ${
                              liveMetrics.pcmMaxIntervalMs?.toFixed(0) ?? "—"
                            } ms · boşluk ${liveMetrics.pcmGapCount ?? 0}`
                      }
                    />
                    <MetadataRow
                      label="İlk kısmi metin gecikmesi (medyan)"
                      value={
                        liveMetrics.partial == null
                          ? "—"
                          : `${liveMetrics.partial.toFixed(2)} sn`
                      }
                    />
                    <MetadataRow
                      label="Tamamlanan metin gecikmesi (medyan)"
                      value={
                        liveMetrics.committed == null
                          ? "—"
                          : `${liveMetrics.committed.toFixed(2)} sn`
                      }
                    />
                    <MetadataRow
                      label="Konuşmacı etiketi gecikmesi (medyan)"
                      value={
                        liveMetrics.label == null ? "—" : `${liveMetrics.label.toFixed(2)} sn`
                      }
                    />
                    <MetadataRow
                      label="Kayan pencere istekleri"
                      value={`${liveMetrics.requests} istek · ${liveMetrics.rollingSeconds.toFixed(1)} sn ses`}
                    />
                  </>
                ) : null}
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

      {renameSpeaker !== null ? (
        <SpeakerRenameDialog
          canonical={renameSpeaker}
          currentName={aliases[renameSpeaker] ?? renameSpeaker}
          busy={renameBusy}
          error={renameError}
          onSave={handleRenameSave}
          onReset={handleRenameReset}
          onCancel={() => {
            setRenameSpeaker(null);
            setRenameError(null);
          }}
        />
      ) : null}
    </section>
  );
}
