/** Meeting-scoped speaker display resolution. One resolver for every surface. */

export type SpeakerAliases = Record<string, string>;

export const UNKNOWN_SPEAKER = "Bilinmeyen";

/**
 * Display name for a canonical speaker label.
 *
 * Aliases are display-only and meeting-local; "Bilinmeyen" is never aliased.
 */
export function resolveSpeakerDisplay(
  canonical: string,
  aliases: SpeakerAliases | null | undefined,
): string {
  if (!aliases || canonical === UNKNOWN_SPEAKER) {
    return canonical;
  }
  const alias = aliases[canonical];
  return alias && alias.trim() ? alias : canonical;
}

export function isCanonicalSpeaker(label: string): boolean {
  return /^Kişi \d{1,2}$/.test(label);
}

/** Stable tone index for a canonical label (Kişi 1 -> 0), -1 when not canonical. */
export function canonicalSpeakerIndex(label: string): number {
  const match = /^Kişi (\d{1,2})$/.exec(label);
  if (!match) return -1;
  return Number(match[1]) - 1;
}

export function applyAlias(
  aliases: SpeakerAliases,
  canonical: string,
  displayName: string | null,
): SpeakerAliases {
  const next = { ...aliases };
  if (displayName === null) {
    delete next[canonical];
  } else {
    next[canonical] = displayName;
  }
  return next;
}
