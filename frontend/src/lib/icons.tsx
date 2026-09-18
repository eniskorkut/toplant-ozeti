/** Inline icon set: one stroke weight, currentColor, never separate assets. */

type IconProps = { className?: string };

function baseProps(className?: string) {
  return {
    "aria-hidden": true as const,
    className,
    fill: "none",
    stroke: "currentColor",
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    strokeWidth: 1.5,
    viewBox: "0 0 24 24",
  };
}

export function MicIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}

export function StopIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
    </svg>
  );
}

export function PlayIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="M8 5.5v13l10-6.5z" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function PauseIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="M9 5.5v13M15 5.5v13" />
    </svg>
  );
}

export function TrashIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="M4 7h16" />
      <path d="M10 11v6M14 11v6" />
      <path d="M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12" />
      <path d="M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

export function CopyIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h9a1 1 0 0 1 1 1v1" />
    </svg>
  );
}

export function CheckIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="m5 12.5 4.5 4.5L19 7" />
    </svg>
  );
}

export function ChevronDownIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="m6 9.5 6 6 6-6" />
    </svg>
  );
}

export function VolumeIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="M11 5.5 6.5 9H4v6h2.5L11 18.5z" />
      <path d="M15 9.5a4 4 0 0 1 0 5" />
    </svg>
  );
}

export function RefreshIcon({ className }: IconProps) {
  return (
    <svg {...baseProps(className)}>
      <path d="M20 11a8 8 0 1 0-2.34 5.66" />
      <path d="M20 5v6h-6" />
    </svg>
  );
}
