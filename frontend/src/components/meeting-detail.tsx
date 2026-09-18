"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  getAnalysis,
  getMeeting,
  getTranscript,
  meetingAudioUrl,
  processMeeting,
  startAnalysis,
  type MeetingAnalysis,
  type MeetingStatus,
  type Transcript,
} from "@/lib/api";
import { formatDuration, formatTimestamp, statusLabel } from "@/lib/format";

const POLL_INTERVAL_MS = 2000;
const PROVIDER_BADGES: Record<string, string> = {
  local: "Yerel · whisper.cpp",
  elevenlabs: "ElevenLabs · Scribe v2",
};

const ANALYSIS_UNAVAILABLE_MESSAGE =
  "Toplantı analizi için uygun bir LLM sağlayıcısı yapılandırılmamış.";

function TimestampButton({
  seconds,
  onSeek,
  label,
}: {
  seconds: number;
  onSeek: (seconds: number) => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onSeek(seconds)}
      className="rounded font-mono text-xs text-sky-700 underline decoration-dotted underline-offset-4 tabular-nums hover:text-sky-900 dark:text-sky-400 dark:hover:text-sky-200"
      aria-label={label ?? `Sesi ${formatTimestamp(seconds)} konumuna getir`}
    >
      {formatTimestamp(seconds)}
    </button>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-4">
      <h3 className="text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
        {title}
      </h3>
      <div className="mt-2">{children}</div>
    </section>
  );
}

export function MeetingDetail({ meetingId }: { meetingId: string }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const [meeting, setMeeting] = useState<MeetingStatus | null>(null);
  const [transcript, setTranscript] = useState<Transcript | null>(null);
  const [analysis, setAnalysis] = useState<MeetingAnalysis | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [analysisBusy, setAnalysisBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);

  const seekTo = useCallback((seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, seconds);
    void audio.play().catch(() => undefined);
  }, []);

  // Initial load: everything is reconstructed from the backend on refresh.
  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      try {
        const status = await getMeeting(meetingId);
        if (cancelled) return;
        setMeeting(status);
        setLoadError(null);

        if (status.status === "completed") {
          const turns = await getTranscript(meetingId);
          if (!cancelled) setTranscript(turns);
        }

        try {
          const analysisBody = await getAnalysis(meetingId);
          if (!cancelled) setAnalysis(analysisBody);
        } catch (analysisLoadError) {
          // No analysis yet is a normal state, not a page error.
          if (!(analysisLoadError instanceof ApiError) || analysisLoadError.status !== 404) {
            if (!cancelled) {
              setAnalysisError(
                analysisLoadError instanceof Error ? analysisLoadError.message : null,
              );
            }
          }
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : "Toplantı yüklenemedi.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [meetingId]);

  // Processing poll: one timer at a time, cleaned up on unmount.
  useEffect(() => {
    if (!meeting || (meeting.status !== "processing" && meeting.status !== "queued")) return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const status = await getMeeting(meetingId);
        if (cancelled) return;
        setMeeting(status);
        if (status.status === "completed") {
          const turns = await getTranscript(meetingId);
          if (!cancelled) setTranscript(turns);
        }
      } catch {
        // Keep the previous state; the next poll retries.
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [meeting, meetingId]);

  // Analysis poll while a job is in flight.
  useEffect(() => {
    if (!analysis || (analysis.status !== "queued" && analysis.status !== "processing")) return;
    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const updated = await getAnalysis(meetingId);
        if (!cancelled) setAnalysis(updated);
      } catch {
        // Keep the previous state; the next poll retries.
      }
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [analysis, meetingId]);

  const handleRetryWithLocal = useCallback(async () => {
    setRetrying(true);
    try {
      // Explicit choice: retry locally. The backend re-queues failed meetings and
      // there is no automatic provider fallback.
      const updated = await processMeeting(meetingId, {
        speakerCount: meeting?.requested_speaker_count ?? null,
        transcriptionProvider: "local",
      });
      setMeeting(updated);
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "Yeniden deneme başarısız.");
    } finally {
      setRetrying(false);
    }
  }, [meeting, meetingId]);

  const handleAnalyze = useCallback(async () => {
    setAnalysisBusy(true);
    setAnalysisError(null);
    try {
      const created = await startAnalysis(meetingId);
      setAnalysis(created);
    } catch (error) {
      if (error instanceof ApiError && error.status === 503) {
        setAnalysisError(ANALYSIS_UNAVAILABLE_MESSAGE);
      } else {
        setAnalysisError(
          error instanceof Error ? error.message : "Analiz başlatılamadı.",
        );
      }
    } finally {
      setAnalysisBusy(false);
    }
  }, [meetingId]);

  if (loading) {
    return <p className="text-sm text-zinc-500 dark:text-zinc-400">Yükleniyor…</p>;
  }

  if (loadError || !meeting) {
    return (
      <p role="alert" className="rounded-lg bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-400">
        {loadError ?? "Toplantı bulunamadı."}
      </p>
    );
  }

  const isProcessing = meeting.status === "queued" || meeting.status === "processing";
  // Legacy meetings may have no provider metadata; render normally without a badge.
  const providerBadge = meeting.transcription_provider
    ? (PROVIDER_BADGES[meeting.transcription_provider] ?? meeting.transcription_provider)
    : null;

  return (
    <div className="space-y-6">
      <header className="surface rounded-2xl p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h1 className="text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            Toplantı {meeting.meeting_id}
          </h1>
          <span className="text-xs text-zinc-500 dark:text-zinc-400">
            {providerBadge ? (
              <span className="mr-2 rounded-full bg-zinc-500/10 px-2 py-0.5 text-zinc-700 dark:text-zinc-300">
                {providerBadge}
              </span>
            ) : null}
            Durum: {statusLabel(meeting.status)}
            {meeting.duration_seconds != null
              ? ` · ${formatDuration(meeting.duration_seconds)}`
              : ""}
          </span>
        </div>

        {isProcessing ? (
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            Ses yazıya dönüştürülüyor ve konuşmacılar ayrılıyor.
          </p>
        ) : null}

        {meeting.status === "failed" ? (
          <div className="mt-2 space-y-2">
            <p
              role="alert"
              className="rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
            >
              İşleme hatası: {meeting.processing_error ?? "bilinmeyen hata"}
            </p>
            <button
              type="button"
              onClick={handleRetryWithLocal}
              disabled={retrying}
              className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
            >
              {retrying ? "Kuyruğa alınıyor…" : "Yerel ile yeniden dene"}
            </button>
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              Yeniden deneme, kaydı yerel işlem hattında işler.
            </p>
          </div>
        ) : null}

        {meeting.status === "completed" ? (
          <audio
            ref={audioRef}
            className="mt-3 w-full"
            controls
            preload="metadata"
            src={meetingAudioUrl(meetingId)}
          />
        ) : null}
      </header>

      {meeting.status === "completed" && transcript ? (
        <section aria-labelledby="transcript" className="surface rounded-2xl p-4">
          <div className="flex items-baseline justify-between gap-4">
            <h2
              id="transcript"
              className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
            >
              Transkript
            </h2>
            <span className="text-xs text-zinc-500 dark:text-zinc-400">
              {transcript.turns.length} konuşma · {transcript.speakers.join(", ")}
            </span>
          </div>

          <ol className="mt-3 space-y-3">
            {transcript.turns.map((turn) => (
              <li key={turn.ordinal} className="flex gap-3">
                <TimestampButton seconds={turn.start_seconds} onSeek={seekTo} />
                <span className="min-w-0 flex-1">
                  <span
                    className={
                      turn.speaker === transcript.unresolved_label
                        ? "block text-xs text-zinc-400 italic dark:text-zinc-500"
                        : "block text-xs font-medium text-zinc-700 dark:text-zinc-300"
                    }
                  >
                    {turn.speaker}
                  </span>
                  <span className="mt-0.5 block text-sm text-zinc-900 dark:text-zinc-100">
                    {turn.text}
                  </span>
                </span>
              </li>
            ))}
          </ol>

          {transcript.unresolved_turns > 0 ? (
            <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
              {transcript.unresolved_turns} bölümde konuşmacı çözümlenemedi (
              {transcript.unresolved_label}).
            </p>
          ) : null}
        </section>
      ) : null}

      {meeting.status === "completed" ? (
        <section aria-labelledby="analysis" className="surface rounded-2xl p-4">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <h2
              id="analysis"
              className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
            >
              Toplantı analizi
            </h2>
            {!analysis || analysis.status === "failed" ? (
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={analysisBusy}
                className="rounded-lg bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-transform duration-160 ease-out active:scale-[0.96] disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900"
              >
                {analysisBusy ? "Başlatılıyor…" : "Toplantıyı Analiz Et"}
              </button>
            ) : (
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                {statusLabel(analysis.status)}
                {analysis.provider ? ` · ${analysis.provider}` : ""}
              </span>
            )}
          </div>

          {analysisError ? (
            <p
              role="status"
              className="mt-3 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300"
            >
              {analysisError}
            </p>
          ) : null}

          {analysis?.analysis_error ? (
            <p
              role="alert"
              className="mt-3 rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
            >
              Analiz hatası: {analysis.analysis_error}
            </p>
          ) : null}

          {analysis && (analysis.status === "queued" || analysis.status === "processing") ? (
            <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
              Analiz sırada; tamamlandığında bu bölüm güncellenir.
            </p>
          ) : null}

          {analysis?.status === "completed" ? (
            <>
              {analysis.summary ? (
                <Section title="Özet">
                  <p className="text-sm text-zinc-900 dark:text-zinc-100">{analysis.summary}</p>
                </Section>
              ) : null}

              {analysis.topics.length > 0 ? (
                <Section title="Konular">
                  <ul className="flex flex-wrap gap-2">
                    {analysis.topics.map((topic) => (
                      <li
                        key={topic}
                        className="rounded-full bg-zinc-500/10 px-2.5 py-1 text-xs text-zinc-700 dark:text-zinc-300"
                      >
                        {topic}
                      </li>
                    ))}
                  </ul>
                </Section>
              ) : null}

              {analysis.decisions.length > 0 ? (
                <Section title="Kararlar">
                  <ul className="space-y-2">
                    {analysis.decisions.map((decision, index) => (
                      <li key={`${decision.text}-${index}`} className="flex gap-3">
                        <TimestampButton seconds={decision.timestamp_seconds} onSeek={seekTo} />
                        <span className="text-sm text-zinc-900 dark:text-zinc-100">
                          {decision.text}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Section>
              ) : null}

              {analysis.action_items.length > 0 ? (
                <Section title="Aksiyonlar">
                  <ul className="space-y-2">
                    {analysis.action_items.map((item, index) => (
                      <li key={`${item.task}-${index}`} className="flex gap-3">
                        <TimestampButton seconds={item.timestamp_seconds} onSeek={seekTo} />
                        <span className="text-sm text-zinc-900 dark:text-zinc-100">
                          {item.task}
                          <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                            {item.owner ?? "Sorumlu belirtilmemiş"}
                            {item.due_date_text ? ` · ${item.due_date_text}` : ""}
                          </span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </Section>
              ) : null}

              {analysis.important_moments.length > 0 ? (
                <Section title="Önemli Anlar">
                  <ul className="space-y-2">
                    {analysis.important_moments.map((moment, index) => (
                      <li key={`${moment.title}-${index}`} className="flex gap-3">
                        <TimestampButton seconds={moment.timestamp_seconds} onSeek={seekTo} />
                        <span className="text-sm text-zinc-900 dark:text-zinc-100">
                          {moment.title}
                          {moment.description ? (
                            <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                              {moment.description}
                            </span>
                          ) : null}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Section>
              ) : null}
            </>
          ) : null}
        </section>
      ) : null}
    </div>
  );
}
