"use client";

import { useState } from "react";

import { ChevronDownIcon } from "@/lib/icons";

/**
 * A collapsible panel used for the meeting detail columns. Native <details> gives
 * keyboard/screen-reader behaviour for free; `open` is mirrored in state so the
 * section can start open yet still be toggled by the user.
 */
export function CollapsibleSection({
  id,
  title,
  meta,
  actions,
  defaultOpen = true,
  children,
}: {
  id?: string;
  title: string;
  meta?: React.ReactNode;
  actions?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <section id={id} className="surface overflow-hidden rounded-2xl">
      <details
        open={open}
        onToggle={(event) => setOpen(event.currentTarget.open)}
        className="group"
      >
        <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-x-4 gap-y-2 p-4 sm:p-5">
          <span className="flex items-center gap-2">
            <ChevronDownIcon className="size-4 shrink-0 text-zinc-500 transition-transform duration-150 ease-out group-open:rotate-180 dark:text-zinc-400" />
            <span className="text-sm font-medium tracking-tight text-zinc-900 dark:text-zinc-100">
              {title}
            </span>
          </span>
          <span className="flex flex-wrap items-center gap-2">
            {meta}
            {actions}
          </span>
        </summary>
        <div className="px-4 pb-4 sm:px-5 sm:pb-5">{children}</div>
      </details>
    </section>
  );
}
