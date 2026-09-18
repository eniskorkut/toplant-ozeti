"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { ApiError, listMeetings, type MeetingSummary } from "@/lib/api";
import { formatDateTime, formatDuration, statusLabel } from "@/lib/format";

const PROVIDER_LABELS: Record<string, string> = {
  local: "Yerel",
  elevenlabs: "ElevenLabs",
};

export function MeetingHistory() {
  const [meetings, setMeetings] = useState<MeetingSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // State is only set from the promise callbacks, never synchronously in the effect.
  useEffect(() => {
    let cancelled = false;
    listMeetings()
      .then((body) => {
        if (cancelled) return;
        setMeetings(body.meetings);
        setError(null);
      })
      .catch((loadError: unknown) => {
        if (cancelled) return;
        setError(loadError instanceof ApiError ? loadError.message : "Toplantılar yüklenemedi.");
      });
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  return (
    <section aria-labelledby="history" className="enter enter-3">
      <div className="mb-3 flex items-baseline justify-between gap-4">
        <h2
          id="history"
          className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
        >
          Geçmiş toplantılar
        </h2>
        <button
          type="button"
          onClick={() => setRefreshKey((value) => value + 1)}
          className="text-xs text-zinc-500 underline underline-offset-4 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-100"
        >
          Yenile
        </button>
      </div>

      {error ? (
        <p role="alert" className="rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400">
          {error}
        </p>
      ) : null}

      {meetings === null && !error ? (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">Yükleniyor…</p>
      ) : null}

      {meetings !== null && meetings.length === 0 ? (
        <div className="surface rounded-2xl p-4">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Henüz toplantı yok. Yukarıdan ilk kaydınızı başlatın.
          </p>
        </div>
      ) : null}

      {meetings !== null && meetings.length > 0 ? (
        <ul className="surface divide-y divide-zinc-950/5 rounded-2xl p-2 dark:divide-white/5">
          {meetings.map((meeting) => (
            <li key={meeting.meeting_id} className="flex items-center gap-3 px-3 py-2.5">
              <span className="min-w-0 flex-1">
                <span className="block text-sm text-zinc-900 dark:text-zinc-100">
                  {formatDateTime(meeting.created_at)}
                </span>
                <span className="block text-xs text-zinc-500 dark:text-zinc-400">
                  {formatDuration(meeting.duration_seconds)} · {statusLabel(meeting.status)}
                  {meeting.analysis_status
                    ? ` · Analiz: ${statusLabel(meeting.analysis_status)}`
                    : ""}
                  {meeting.requested_speaker_count
                    ? ` · ${meeting.requested_speaker_count} konuşmacı`
                    : ""}
                  {meeting.transcription_provider
                    ? ` · ${PROVIDER_LABELS[meeting.transcription_provider] ?? meeting.transcription_provider}`
                    : ""}
                </span>
              </span>
              <Link
                href={`/meetings/${meeting.meeting_id}`}
                className="shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium text-zinc-700 underline underline-offset-4 hover:text-zinc-950 dark:text-zinc-300 dark:hover:text-zinc-50"
              >
                Aç
              </Link>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
