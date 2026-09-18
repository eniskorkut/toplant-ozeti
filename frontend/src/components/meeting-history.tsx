"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { ConfirmDialog } from "@/components/confirm-dialog";
import { Chip, statusTone, StatusChip } from "@/components/chips";
import { ApiError, deleteMeeting, listMeetings, type MeetingSummary } from "@/lib/api";
import { formatDuration, formatMeetingDateShort, statusLabel } from "@/lib/format";
import { RefreshIcon, TrashIcon } from "@/lib/icons";

const PROVIDER_LABELS: Record<string, string> = {
  local: "Yerel",
  elevenlabs: "ElevenLabs",
};

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

  const meetingCount = meetings?.length ?? 0;

  return (
    <section aria-labelledby="history" className="enter enter-3">
      <div className="mb-3 flex items-baseline justify-between gap-4">
        <h2
          id="history"
          className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
        >
          Geçmiş toplantılar
          {meetings !== null && meetingCount > 0 ? (
            <span className="ml-2 text-xs font-normal text-zinc-500 dark:text-zinc-400">
              {meetingCount}
            </span>
          ) : null}
        </h2>
        <button
          type="button"
          onClick={() => setRefreshKey((value) => value + 1)}
          className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-zinc-500 transition-colors duration-150 ease-out hover:bg-zinc-500/10 hover:text-zinc-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:text-zinc-400 dark:hover:text-zinc-100"
        >
          <RefreshIcon className="size-3.5" />
          Yenile
        </button>
      </div>

      {error ? (
        <p
          role="alert"
          className="rounded-xl bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
        >
          {error}
        </p>
      ) : null}

      {notice ? (
        <p
          role="status"
          aria-live="polite"
          className="mb-2 rounded-xl bg-emerald-500/10 px-3 py-2 text-xs text-emerald-800 dark:text-emerald-300"
        >
          {notice}
        </p>
      ) : null}

      {meetings === null && !error ? (
        <p className="text-xs text-zinc-500 dark:text-zinc-400">Yükleniyor…</p>
      ) : null}

      {meetings !== null && meetingCount === 0 ? (
        <div className="surface rounded-2xl p-5">
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            Henüz toplantı yok. Yukarıdan ilk kaydınızı başlatın.
          </p>
        </div>
      ) : null}

      {meetings !== null && meetingCount > 0 ? (
        <ul className="surface divide-y divide-zinc-950/5 rounded-2xl p-2 dark:divide-white/5">
          {meetings.map((meeting) => (
            <li
              key={meeting.meeting_id}
              className="flex items-center gap-2 rounded-xl px-3 py-3 transition-colors duration-150 ease-out hover:bg-zinc-500/5 sm:gap-3"
            >
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                  <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                    {formatMeetingDateShort(meeting.created_at)}
                  </span>
                  <span className="font-mono text-xs tabular-nums text-zinc-500 dark:text-zinc-400">
                    {formatDuration(meeting.duration_seconds)}
                  </span>
                </div>
                <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                  <StatusChip status={meeting.status} label={statusLabel(meeting.status)} />
                  {meeting.has_transcript && meeting.transcription_provider ? (
                    <Chip>
                      {PROVIDER_LABELS[meeting.transcription_provider] ??
                        meeting.transcription_provider}
                    </Chip>
                  ) : null}
                  {meeting.analysis_status ? (
                    <Chip tone={statusTone(meeting.analysis_status)}>
                      Analiz: {statusLabel(meeting.analysis_status)}
                    </Chip>
                  ) : null}
                  {meeting.requested_speaker_count ? (
                    <Chip>{meeting.requested_speaker_count} konuşmacı</Chip>
                  ) : null}
                  <span className="hidden font-mono text-[11px] text-zinc-400 sm:inline dark:text-zinc-500">
                    {meeting.meeting_id.slice(0, 8)}
                  </span>
                </div>
              </div>

              <Link
                href={`/meetings/${meeting.meeting_id}`}
                className="shrink-0 rounded-xl px-3 py-1.5 text-xs font-medium text-zinc-700 transition-colors duration-150 ease-out hover:bg-zinc-500/10 hover:text-zinc-950 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:text-zinc-300 dark:hover:text-zinc-50"
              >
                Aç
              </Link>
              <button
                type="button"
                onClick={() => {
                  setDeleteError(null);
                  setPendingDelete(meeting);
                }}
                aria-label={`${formatMeetingDateShort(meeting.created_at)} · Toplantıyı sil`}
                className="shrink-0 rounded-xl p-2 text-zinc-500 transition-colors duration-150 ease-out hover:bg-red-500/10 hover:text-red-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:text-zinc-400 dark:hover:text-red-400"
              >
                <TrashIcon className="size-4" />
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
