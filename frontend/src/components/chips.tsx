const STATUS_TONES: Record<string, string> = {
  completed: "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400",
  failed: "bg-red-500/10 text-red-700 dark:text-red-400",
  processing: "bg-amber-500/10 text-amber-800 dark:text-amber-300",
  queued: "bg-amber-500/10 text-amber-800 dark:text-amber-300",
  uploaded: "bg-zinc-500/10 text-zinc-600 dark:text-zinc-400",
};

const NEUTRAL_TONE = "bg-zinc-500/10 text-zinc-600 dark:text-zinc-400";

export function statusTone(status: string): string {
  return STATUS_TONES[status] ?? NEUTRAL_TONE;
}

export function Chip({
  tone = NEUTRAL_TONE,
  children,
}: {
  tone?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium ${tone}`}
    >
      {children}
    </span>
  );
}

export function StatusChip({ status, label }: { status: string; label: string }) {
  return <Chip tone={statusTone(status)}>{label}</Chip>;
}
