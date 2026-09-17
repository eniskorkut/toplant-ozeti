/**
 * Browser capability detection for future microphone recording support.
 *
 * This module is intentionally side-effect free: it never requests microphone
 * permission and never constructs a MediaRecorder. It only reports what the
 * current browser can do so the UI can enable recording once it is implemented.
 */

export type MediaCapabilities = {
  /** navigator.mediaDevices is available (requires a secure context). */
  mediaDevices: boolean;
  /** navigator.mediaDevices.getUserMedia is available. */
  getUserMedia: boolean;
  /** window.MediaRecorder is available. */
  mediaRecorder: boolean;
  /** A MIME type supported by MediaRecorder was found. */
  supportedMimeType: string | null;
};

/**
 * Candidate recording formats, most preferred first.
 * Opus in WebM is the best-supported combination across Chromium and Firefox;
 * Safari falls back to audio/mp4 (AAC).
 */
export const RECORDING_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
] as const;

export type RecordingMimeType = (typeof RECORDING_MIME_CANDIDATES)[number];

function hasMediaDevices(): boolean {
  return typeof navigator !== "undefined" && Boolean(navigator.mediaDevices);
}

export function isGetUserMediaSupported(): boolean {
  return hasMediaDevices() && typeof navigator.mediaDevices.getUserMedia === "function";
}

export function isMediaRecorderSupported(): boolean {
  return typeof window !== "undefined" && typeof window.MediaRecorder === "function";
}

/**
 * Returns the first MediaRecorder MIME type the browser reports as supported,
 * or null when none of the candidates (or MediaRecorder itself) is available.
 */
export function getPreferredRecordingMimeType(): RecordingMimeType | null {
  if (typeof window === "undefined" || typeof window.MediaRecorder !== "function") {
    return null;
  }

  const { isTypeSupported } = window.MediaRecorder;
  if (typeof isTypeSupported !== "function") {
    return null;
  }

  for (const mimeType of RECORDING_MIME_CANDIDATES) {
    if (isTypeSupported(mimeType)) {
      return mimeType;
    }
  }

  return null;
}

export function getMediaCapabilities(): MediaCapabilities {
  return {
    mediaDevices: hasMediaDevices(),
    getUserMedia: isGetUserMediaSupported(),
    mediaRecorder: isMediaRecorderSupported(),
    supportedMimeType: getPreferredRecordingMimeType(),
  };
}

/** True when the browser can capture and record audio. */
export function canRecordAudio(): boolean {
  const capabilities = getMediaCapabilities();
  return capabilities.getUserMedia && capabilities.mediaRecorder;
}
