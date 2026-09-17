"use client";

import { useSyncExternalStore } from "react";

import {
  getMediaCapabilities,
  RECORDING_MIME_CANDIDATES,
  type MediaCapabilities,
} from "@/lib/browser-media";

type Status = "checking" | "ready" | "unavailable";

type CapabilityRow = {
  label: string;
  detail: string;
  status: Status;
};

function CheckIcon() {
  return (
    <svg
      aria-hidden="true"
      className="size-4 shrink-0 text-emerald-600 dark:text-emerald-500"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="m8.5 12.5 2.5 2.5 4.5-5" />
    </svg>
  );
}

function DashIcon() {
  return (
    <svg
      aria-hidden="true"
      className="size-4 shrink-0 text-zinc-400 dark:text-zinc-600"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <circle cx="12" cy="12" r="9" />
      <path d="M9 12h6" />
    </svg>
  );
}

function PendingIcon() {
  return (
    <svg
      aria-hidden="true"
      className="size-4 shrink-0 animate-pulse text-zinc-300 dark:text-zinc-700"
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.5"
      viewBox="0 0 24 24"
    >
      <circle cx="12" cy="12" r="9" />
    </svg>
  );
}

function StatusIcon({ status }: { status: Status }) {
  if (status === "checking") return <PendingIcon />;
  if (status === "ready") return <CheckIcon />;
  return <DashIcon />;
}

const statusLabels: Record<Status, string> = {
  checking: "Checking…",
  ready: "Supported",
  unavailable: "Unavailable",
};

function statusFor(supported: boolean): Status {
  return supported ? "ready" : "unavailable";
}

// Capabilities never change while the page is open, so there is nothing to
// subscribe to. useSyncExternalStore keeps the client-only read out of render
// and returns null during SSR to avoid a hydration mismatch.
const subscribeToNothing = () => () => {};

let cachedCapabilities: MediaCapabilities | null = null;

function getCapabilitiesSnapshot(): MediaCapabilities {
  cachedCapabilities ??= getMediaCapabilities();
  return cachedCapabilities;
}

function getServerCapabilitiesSnapshot(): null {
  return null;
}

export function BrowserCapabilities() {
  // Detection only: no permission prompt and no MediaRecorder is created.
  const capabilities = useSyncExternalStore(
    subscribeToNothing,
    getCapabilitiesSnapshot,
    getServerCapabilitiesSnapshot,
  );

  const rows: CapabilityRow[] = [
    {
      label: "Microphone access",
      detail: "navigator.mediaDevices.getUserMedia",
      status: capabilities ? statusFor(capabilities.getUserMedia) : "checking",
    },
    {
      label: "MediaRecorder",
      detail: "window.MediaRecorder",
      status: capabilities ? statusFor(capabilities.mediaRecorder) : "checking",
    },
    {
      label: "Preferred format",
      detail: capabilities?.supportedMimeType ?? RECORDING_MIME_CANDIDATES[0],
      status: capabilities ? statusFor(Boolean(capabilities.supportedMimeType)) : "checking",
    },
  ];

  return (
    <section aria-labelledby="browser-capabilities" className="enter enter-3">
      <div className="mb-3 flex items-baseline justify-between gap-4">
        <h2
          id="browser-capabilities"
          className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100"
        >
          Browser capabilities
        </h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">No permission requested</p>
      </div>

      <div className="surface rounded-2xl p-2">
        <ul className="divide-y divide-zinc-950/5 dark:divide-white/5">
          {rows.map((row) => (
            <li key={row.label} className="flex items-center gap-3 px-3 py-2.5">
              <StatusIcon status={row.status} />
              <span className="min-w-0 flex-1">
                <span className="block text-sm text-zinc-900 dark:text-zinc-100">
                  {row.label}
                </span>
                <span className="block truncate font-mono text-xs text-zinc-500 tabular-nums dark:text-zinc-400">
                  {row.detail}
                </span>
              </span>
              <span className="text-xs text-zinc-500 dark:text-zinc-400">
                {statusLabels[row.status]}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
