"use client";

import { useEffect, useId, useRef, useState } from "react";

const FOCUSABLE_SELECTOR =
  'button:not([disabled]), [href], input:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Accessible inline rename dialog for meeting-local speaker aliases.
 * Save requires a trimmed non-empty name; reset restores the canonical label.
 */
export function SpeakerRenameDialog({
  canonical,
  currentName,
  busy = false,
  error = null,
  onSave,
  onReset,
  onCancel,
}: {
  canonical: string;
  currentName: string;
  busy?: boolean;
  error?: string | null;
  onSave: (displayName: string) => void;
  onReset: () => void;
  onCancel: () => void;
}) {
  const [value, setValue] = useState(currentName === canonical ? "" : currentName);
  const dialogRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const errorId = useId();

  useEffect(() => {
    returnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const timer = window.setTimeout(() => inputRef.current?.focus(), 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (!busy) onCancel();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR);
      if (!focusable || focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("keydown", handleKeyDown);
      returnFocusRef.current?.focus();
    };
  }, [busy, onCancel]);

  const trimmed = value.trim();
  const canSave = trimmed.length > 0 && trimmed.length <= 50 && trimmed !== currentName;

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center p-4 sm:items-center">
      <div className="absolute inset-0 bg-zinc-950/45" aria-hidden="true" />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={error ? errorId : undefined}
        className="dialog-enter surface relative w-full max-w-sm rounded-2xl p-5"
      >
        <h2 id={titleId} className="text-base font-semibold tracking-tight">
          Konuşmacı adı
        </h2>
        <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          Yalnızca bu toplantıda görünür. Yeni toplantılar {canonical} ile başlar.
        </p>

        <label htmlFor="speaker-name" className="mt-4 block text-xs font-medium text-zinc-700 dark:text-zinc-300">
          Ad
        </label>
        <input
          id="speaker-name"
          ref={inputRef}
          value={value}
          maxLength={50}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && canSave && !busy) {
              event.preventDefault();
              onSave(trimmed);
            }
          }}
          placeholder={canonical}
          className="mt-1 w-full rounded-xl border border-zinc-950/10 bg-transparent px-3 py-2 text-sm text-zinc-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 dark:border-white/15 dark:text-zinc-100"
        />

        {error ? (
          <p
            id={errorId}
            role="alert"
            className="mt-3 rounded-xl bg-red-500/10 px-3 py-2 text-xs text-red-700 dark:text-red-400"
          >
            {error}
          </p>
        ) : null}

        <div className="mt-5 flex flex-wrap justify-end gap-2">
          {currentName !== canonical ? (
            <button
              type="button"
              onClick={onReset}
              disabled={busy}
              className="mr-auto rounded-xl px-3 py-2 text-sm font-medium text-zinc-600 transition-colors duration-150 ease-out hover:bg-zinc-500/10 hover:text-zinc-900 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:opacity-50 dark:text-zinc-300 dark:hover:text-zinc-50"
            >
              {canonical}&apos;a sıfırla
            </button>
          ) : null}
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="rounded-xl px-3.5 py-2 text-sm font-medium text-zinc-700 transition-colors duration-150 ease-out hover:bg-zinc-500/10 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:opacity-50 dark:text-zinc-200"
          >
            İptal
          </button>
          <button
            type="button"
            onClick={() => canSave && onSave(trimmed)}
            disabled={busy || !canSave}
            className="rounded-xl bg-zinc-900 px-3.5 py-2 text-sm font-medium text-white transition-colors duration-150 ease-out hover:bg-zinc-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-zinc-500 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-white"
          >
            {busy ? "Kaydediliyor…" : "Kaydet"}
          </button>
        </div>
      </div>
    </div>
  );
}
