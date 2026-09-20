"use client";

import { useEffect, useRef } from "react";

import { formatTimestamp } from "@/lib/format";
import { canonicalSpeakerIndex, resolveSpeakerDisplay, type SpeakerAliases } from "@/lib/speakers";

export type LiveLine = {
  id: number;
  startSeconds: number;
  endSeconds: number;
  text: string;
  /** Canonical label once rolling diarization has attached one. */
  canonical: string | null;
  provisional: boolean;
};

const SPEAKER_TONES = [
  "bg-sky-500/10 text-sky-700 dark:text-sky-300",
  "bg-violet-500/10 text-violet-700 dark:text-violet-300",
  "bg-teal-500/10 text-teal-700 dark:text-teal-300",
  "bg-amber-500/10 text-amber-800 dark:text-amber-300",
  "bg-rose-500/10 text-rose-700 dark:text-rose-300",
  "bg-indigo-500/10 text-indigo-700 dark:text-indigo-300",
];

const NEAR_BOTTOM_PX = 48;

export function LiveTranscript({
  lines,
  aliases,
  status,
  warning,
  onRenameSpeaker,
}: {
  lines: LiveLine[];
  aliases: SpeakerAliases;
  status: string;
  warning: string | null;
  onRenameSpeaker: (canonical: string) => void;
}) {
  const listRef = useRef<HTMLOListElement | null>(null);
  const nearBottomRef = useRef(true);

  // Always render committed utterances chronologically (late speaker patches must
  // never reinsert or reorder them); the newest provisional line stays at the end.
  const committed = lines
    .filter((line) => !line.provisional)
    .sort((left, right) => left.startSeconds - right.startSeconds || left.id - right.id);
  const provisional = lines.filter((line) => line.provisional);
  const ordered = [...committed, ...provisional];

  // Auto-scroll only while the user is already near the bottom.
  useEffect(() => {
    const list = listRef.current;
    if (!list || !nearBottomRef.current) return;
    list.scrollTop = list.scrollHeight;
  }, [lines]);

  return (
    <div className="mt-4 rounded-xl bg-zinc-500/5 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-baseline gap-2">
          <h3 className="text-xs font-medium tracking-wide text-zinc-500 uppercase dark:text-zinc-400">
            Canlı transkript
          </h3>
          <span className="text-[11px] text-zinc-400 dark:text-zinc-500">
            · Konuşmacı etiketleri kayıt sırasında tahminidir.
          </span>
        </div>
        <span className="inline-flex items-center gap-1.5 text-[11px] text-zinc-500 dark:text-zinc-400">
          <span
            className={`size-1.5 rounded-full ${status === "Dinleniyor" ? "bg-red-500" : "bg-zinc-400"}`}
            aria-hidden="true"
          />
          {status}
        </span>
      </div>

      {warning ? (
        <p
          role="status"
          className="mt-2 rounded-lg bg-amber-500/10 px-2.5 py-1.5 text-[11px] text-amber-800 dark:text-amber-300"
        >
          {warning}
        </p>
      ) : null}

      <ol
        ref={listRef}
        onScroll={(event) => {
          const element = event.currentTarget;
          nearBottomRef.current =
            element.scrollHeight - element.scrollTop - element.clientHeight < NEAR_BOTTOM_PX;
        }}
        className="mt-2 max-h-72 space-y-1 overflow-y-auto pr-1"
      >
        {ordered.length === 0 ? (
          <li className="px-2 py-3 text-xs text-zinc-500 dark:text-zinc-400">
            Konuşma bekleniyor…
          </li>
        ) : null}
        {ordered.map((line) => {
          const toneIndex = line.canonical ? canonicalSpeakerIndex(line.canonical) : -1;
          const tone = toneIndex >= 0 ? SPEAKER_TONES[toneIndex % SPEAKER_TONES.length] : "";
          const display = line.canonical
            ? resolveSpeakerDisplay(line.canonical, aliases)
            : null;
          return (
            <li
              key={line.id}
              className={`flex gap-3 rounded-lg px-2 py-1.5 ${
                line.provisional ? "opacity-60" : ""
              }`}
            >
              <span className="shrink-0 font-mono text-[11px] text-zinc-500 tabular-nums dark:text-zinc-400">
                {formatTimestamp(line.startSeconds)}
              </span>
              <span className="min-w-0 flex-1">
                {line.canonical ? (
                  <button
                    type="button"
                    onClick={() => onRenameSpeaker(line.canonical as string)}
                    title="Canlı konuşmacı etiketi (tahmini)"
                    aria-label={`${display} · Konuşmacı adını düzenle`}
                    className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium transition-colors duration-150 ease-out hover:brightness-95 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 ${tone}`}
                  >
                    {display}
                    <svg
                      aria-hidden="true"
                      className="size-3 opacity-60"
                      fill="none"
                      stroke="currentColor"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="1.5"
                      viewBox="0 0 24 24"
                    >
                      <path d="M4 20h4L19 9l-4-4L4 16z" />
                      <path d="M14 6l4 4" />
                    </svg>
                  </button>
                ) : (
                  <span className="text-[11px] text-zinc-400 italic dark:text-zinc-500">
                    Konuşmacı belirleniyor
                  </span>
                )}
                <span className="mt-0.5 block text-sm leading-relaxed text-zinc-900 dark:text-zinc-100">
                  {line.text}
                </span>
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
