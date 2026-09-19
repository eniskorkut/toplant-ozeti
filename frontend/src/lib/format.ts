/** Small formatting helpers shared by the meeting UI. */

/**
 * Display timezone is pinned to Istanbul: the product is Turkish and meeting
 * times must read the same regardless of the viewer's device timezone.
 */
export const DISPLAY_TIME_ZONE = "Europe/Istanbul";

/**
 * Timestamps without an explicit offset are stored UTC (SQLite rows) and must
 * not be re-interpreted as device-local time. The API now appends "+00:00";
 * this is the defensive fallback for cached or legacy payloads.
 */
function parseTimestamp(isoTimestamp: string): Date {
  const hasOffset = /(?:z|[+-]\d{2}:?\d{2})$/i.test(isoTimestamp.trim());
  return new Date(hasOffset ? isoTimestamp : `${isoTimestamp}Z`);
}

export function formatDuration(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null || Number.isNaN(totalSeconds)) {
    return "—";
  }
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  }
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

/** Transcript/analysis timestamp, e.g. 42.35 -> "00:42" (hours only when needed). */
export function formatTimestamp(totalSeconds: number | null | undefined): string {
  if (totalSeconds == null || Number.isNaN(totalSeconds)) {
    return "--:--";
  }
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const rest = seconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

export function formatDateTime(isoTimestamp: string): string {
  const date = parseTimestamp(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return isoTimestamp;
  }
  return date.toLocaleString("tr-TR", {
    timeZone: DISPLAY_TIME_ZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Human detail header, e.g. "18 Eylül 2026, 20:03". */
export function formatMeetingDate(isoTimestamp: string): string {
  const date = parseTimestamp(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return isoTimestamp;
  }
  return date.toLocaleString("tr-TR", {
    timeZone: DISPLAY_TIME_ZONE,
    day: "numeric",
    month: "long",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Compact history row label, e.g. "18 Eyl 2026, 20:03". */
export function formatMeetingDateShort(isoTimestamp: string): string {
  const date = parseTimestamp(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return isoTimestamp;
  }
  return date.toLocaleString("tr-TR", {
    timeZone: DISPLAY_TIME_ZONE,
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export const STATUS_LABELS: Record<string, string> = {
  uploaded: "Yüklendi",
  queued: "Sırada",
  processing: "İşleniyor",
  completed: "Tamamlandı",
  failed: "Başarısız",
};

export function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status;
}
