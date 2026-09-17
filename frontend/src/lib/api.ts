const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (!configured) {
    return DEFAULT_API_BASE_URL;
  }
  return configured.replace(/\/+$/, "");
}

export type RecordingInput = {
  mime_type: string;
  size_bytes: number;
};

export type RecordingMp3 = {
  size_bytes: number;
};

export type RecordingProcessingWav = {
  size_bytes: number;
  sample_rate: number;
  channels: number;
};

export type RecordingCreated = {
  recording_id: string;
  duration_seconds: number;
  input: RecordingInput;
  mp3: RecordingMp3;
  processing_wav: RecordingProcessingWav;
  conversion_ms: number;
};

export type UploadRecordingOptions = {
  mimeType: string;
  durationSeconds: number;
  filename: string;
};

async function readErrorMessage(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail.length > 0) {
        return detail;
      }
    }
  } catch {
    // fall through to the generic message below
  }
  return `Upload failed (HTTP ${response.status})`;
}

export async function uploadRecording(
  blob: Blob,
  options: UploadRecordingOptions,
): Promise<RecordingCreated> {
  const form = new FormData();
  form.append("audio", blob, options.filename);
  form.append("mime_type", options.mimeType);
  form.append("client_duration_seconds", options.durationSeconds.toFixed(3));

  const response = await fetch(`${getApiBaseUrl()}/api/recordings`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return (await response.json()) as RecordingCreated;
}
