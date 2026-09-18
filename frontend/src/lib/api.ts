const DEFAULT_API_BASE_URL = "http://localhost:8000";

export function getApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (!configured) {
    return DEFAULT_API_BASE_URL;
  }
  return configured.replace(/\/+$/, "");
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

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
  return `İstek başarısız oldu (HTTP ${response.status})`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, init);
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
  return (await response.json()) as T;
}

// --- recordings ------------------------------------------------------------

export type RecordingInput = { mime_type: string; size_bytes: number };
export type RecordingMp3 = { size_bytes: number };
export type RecordingProcessingWav = {
  size_bytes: number;
  sample_rate: number;
  channels: number;
};
export type RecordingCreated = {
  recording_id: string;
  meeting_id: string;
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

export async function uploadRecording(
  blob: Blob,
  options: UploadRecordingOptions,
): Promise<RecordingCreated> {
  const form = new FormData();
  form.append("audio", blob, options.filename);
  form.append("mime_type", options.mimeType);
  form.append("client_duration_seconds", options.durationSeconds.toFixed(3));
  return request<RecordingCreated>("/api/recordings", { method: "POST", body: form });
}

// --- meetings --------------------------------------------------------------

export type MeetingStatusValue = "uploaded" | "queued" | "processing" | "completed" | "failed";

export type TranscriptionProviderId = "local" | "elevenlabs";

export type ProviderCapability = {
  id: TranscriptionProviderId;
  available: boolean;
  cloud: boolean;
  label: string;
};

export type ProviderCapabilities = {
  default: TranscriptionProviderId;
  providers: ProviderCapability[];
};

export type MeetingStatus = {
  meeting_id: string;
  status: MeetingStatusValue;
  created_at: string;
  duration_seconds: number | null;
  requested_speaker_count: number | null;
  processing_error: string | null;
  has_transcript: boolean;
  transcription_provider: string | null;
  transcription_model: string | null;
};

export type TranscriptTurn = {
  ordinal: number;
  speaker: string;
  start_seconds: number;
  end_seconds: number;
  text: string;
};

export type Transcript = {
  meeting_id: string;
  status: string;
  duration_seconds: number | null;
  speakers: string[];
  unresolved_label: string;
  unresolved_turns: number;
  turns: TranscriptTurn[];
};

export type AnalysisStatusValue = "queued" | "processing" | "completed" | "failed";

export type AnalysisDecision = {
  text: string;
  source_turn_ordinals: number[];
  timestamp_seconds: number;
};

export type AnalysisActionItem = {
  task: string;
  owner: string | null;
  due_date_text: string | null;
  source_turn_ordinals: number[];
  timestamp_seconds: number;
};

export type AnalysisImportantMoment = {
  title: string;
  description: string;
  source_turn_ordinal: number;
  timestamp_seconds: number;
};

export type MeetingAnalysis = {
  meeting_id: string;
  status: AnalysisStatusValue;
  provider: string | null;
  model: string | null;
  summary: string | null;
  topics: string[];
  decisions: AnalysisDecision[];
  action_items: AnalysisActionItem[];
  important_moments: AnalysisImportantMoment[];
  analysis_error: string | null;
  input_chars: number | null;
  latency_seconds: number | null;
  repair_attempts: number;
};

export type MeetingSummary = {
  meeting_id: string;
  created_at: string;
  duration_seconds: number | null;
  status: MeetingStatusValue;
  requested_speaker_count: number | null;
  has_transcript: boolean;
  analysis_status: AnalysisStatusValue | null;
  transcription_provider: string | null;
  transcription_model: string | null;
};

export type MeetingList = { meetings: MeetingSummary[]; count: number };

export function listMeetings(): Promise<MeetingList> {
  return request<MeetingList>("/api/v1/meetings");
}

/** Deletes a meeting, its transcript/analysis and unreferenced artifacts. */
export async function deleteMeeting(meetingId: string): Promise<void> {
  const response = await fetch(`${getApiBaseUrl()}/api/v1/meetings/${meetingId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
}

export function getMeeting(meetingId: string): Promise<MeetingStatus> {
  return request<MeetingStatus>(`/api/v1/meetings/${meetingId}`);
}

export function getTranscript(meetingId: string): Promise<Transcript> {
  return request<Transcript>(`/api/v1/meetings/${meetingId}/transcript`);
}

export type ProcessMeetingOptions = {
  speakerCount: number | null;
  /** null uses the configured server default provider. */
  transcriptionProvider: TranscriptionProviderId | null;
};

export function processMeeting(
  meetingId: string,
  options: ProcessMeetingOptions,
): Promise<MeetingStatus> {
  return request<MeetingStatus>(`/api/v1/meetings/${meetingId}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      speaker_count: options.speakerCount,
      transcription_provider: options.transcriptionProvider,
    }),
  });
}

export function getTranscriptionProviders(): Promise<ProviderCapabilities> {
  return request<ProviderCapabilities>("/api/v1/transcription/providers");
}

export type ElevenLabsUsage = {
  available: boolean;
  tier?: string;
  status?: string;
  usage?: number;
  limit?: number;
  remaining?: number;
  reset_at?: string;
  reason?: string;
};

export function getElevenLabsUsage(): Promise<ElevenLabsUsage> {
  return request<ElevenLabsUsage>("/api/v1/transcription/providers/elevenlabs/usage");
}

export function getAnalysis(meetingId: string): Promise<MeetingAnalysis> {
  return request<MeetingAnalysis>(`/api/v1/meetings/${meetingId}/analysis`);
}

export function startAnalysis(meetingId: string): Promise<MeetingAnalysis> {
  return request<MeetingAnalysis>(`/api/v1/meetings/${meetingId}/analyze`, {
    method: "POST",
  });
}

export function meetingAudioUrl(meetingId: string): string {
  return `${getApiBaseUrl()}/api/v1/meetings/${meetingId}/audio`;
}
