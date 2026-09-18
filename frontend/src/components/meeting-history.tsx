"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { ApiError, deleteMeeting, listMeetings, type MeetingSummary } from "@/lib/api";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { formatDateTime, formatDuration, statusLabel } from "@/lib/format";

const PROVIDER_LABELS: Record<string, string> = {
  local: "Yerel",
  elevenlabs: "ElevenLabs",
};

function TrashIcon() {
  return (
    <svg
      aria-hidden="true"
      className="size-4"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <path d="M4 7h16" />
      <path d="M10 11v6M14 11v6" />
      <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
      <path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

export function MeetingHistory() {
  const [meetings, setMeetings] = useState<MeetingSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [pendingDelete, setPendingDelete] = useState<MeetingSummary | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

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

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const removeMeeting = useCallback((meetingId: string) => {
    setMeetings((current) =>
      current ? current.filter((meeting) => meeting.meeting_id !== meetingId) : current,
    );
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    if (!pendingDelete) return;
    const { meeting_id: meetingId } = pendingDelete;
    setDeleteBusy(true);
    setDeleteError(null);
    try {
      await deleteMeeting(meetingId);
      removeMeeting(meetingId);
      setNotice("Toplantı silindi.");
      setPendingDelete(null);
    } catch (deleteFailure) {
      if (deleteFailure instanceof ApiError && deleteFailure.status === 404) {
        // Already gone elsewhere: treat as success and refresh the visible list.
        removeMeeting(meetingId);
        setNotice("Toplantı silindi.");
        setPendingDelete(null);
      } else if (deleteFailure instanceof ApiError && deleteFailure.status === 409) {
        setDeleteError(
          "Toplantı şu anda işleniyor veya analiz ediliyor; işlem bitince tekrar deneyin.",
        );
      } else {
        setDeleteError(
          deleteFailure instanceof Error ? deleteFailure.message : "Toplantı silinemedi.",
        );
      }
    } finally {
      setDeleteBusy(false);
    }
  }, [pendingDelete, removeMeeting]);

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

      {notice ? (
        <p
          role="status"
          aria-live="polite"
          className="mb-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-xs text-emerald-800 dark:text-emerald-300"
        >
          {notice}
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
              <button
                type="button"
                onClick={() => {
                  setDeleteError(null);
                  setPendingDelete(meeting);
                }}
                aria-label={`${formatDateTime(meeting.created_at)} · Toplantıyı sil`}
                className="shrink-0 rounded-lg p-2 text-zinc-500 transition-colors duration-150 ease-out hover:bg-red-500/10 hover:text-red-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:text-zinc-400 dark:hover:text-red-400"
              >
                <TrashIcon />
              </button>
            </li>
          ))}
        </ul>
      ) : null}

      <ConfirmDialog
        open={pendingDelete !== null}
        title="Bu toplantı silinsin mi?"
        description="Ses kaydı, transkript ve analiz verileri kalıcı olarak silinecek. Bu işlem geri alınamaz."
        confirmLabel="Toplantıyı Sil"
        cancelLabel="İptal"
        busy={deleteBusy}
        error={deleteError}
        onConfirm={handleConfirmDelete}
        onCancel={() => {
          setPendingDelete(null);
          setDeleteError(null);
        }}
      />
    </section>
  );
}
