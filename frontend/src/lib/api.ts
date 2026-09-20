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
  /** When present, live-session aliases/timelines migrate onto the new meeting. */
  liveSessionId?: string | null;
};

export async function uploadRecording(
  blob: Blob,
  options: UploadRecordingOptions,
): Promise<RecordingCreated> {
  const form = new FormData();
  form.append("audio", blob, options.filename);
  form.append("mime_type", options.mimeType);
  form.append("client_duration_seconds", options.durationSeconds.toFixed(3));
  if (options.liveSessionId) {
    form.append("live_session_id", options.liveSessionId);
  }
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

// --- live transcription (ElevenLabs) ----------------------------------------

export type LiveSpeakerState = {
  canonical_speaker: string;
  display_name: string;
  speech_seconds: number;
};

export type LiveSessionState = {
  live_session_id: string;
  windows_received: number;
  rolling_seconds: number;
  label_switches: number;
  speakers: LiveSpeakerState[];
  aliases: Record<string, string>;
};

export type RealtimeToken = { token: string };

export async function getRealtimeToken(): Promise<RealtimeToken> {
  return request<RealtimeToken>("/api/v1/transcription/providers/elevenlabs/realtime-token", {
    method: "POST",
  });
}

export async function createLiveSession(
  speakerCount: number | null = null,
): Promise<{ live_session_id: string }> {
  const query = speakerCount !== null ? `?speaker_count=${speakerCount}` : "";
  return request<{ live_session_id: string }>(
    `/api/v1/live-transcription/sessions${query}`,
    { method: "POST" },
  );
}

export async function deleteLiveSession(liveSessionId: string): Promise<void> {
  const response = await fetch(
    `${getApiBaseUrl()}/api/v1/live-transcription/sessions/${liveSessionId}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
}

export async function getLiveSession(liveSessionId: string): Promise<LiveSessionState> {
  return request<LiveSessionState>(`/api/v1/live-transcription/sessions/${liveSessionId}`);
}

export type SpeakerAssignment = {
  canonical_speaker: string | null;
  speaker_state?: "pending" | "temporally_confirmed" | "final";
  is_new: boolean;
  confidence: number | null;
  evidence: string;
  /** Newest mutable tail: may not label the transcript yet. */
  provisional: boolean;
  start: number;
  end: number;
  speech_seconds: number;
};

export type SpeakerWindowResult = {
  sequence: number;
  window: [number, number];
  assignments: SpeakerAssignment[];
  new_speakers: string[];
  promoted_speakers: string[];
  ambiguous_speakers: number;
  candidate_speakers: number;
  confirmed_speakers: string[];
  provider_speakers: number;
  latency_seconds: number;
  rolling_seconds: number;
  label_switches: number;
};

export type SpeakerWindowRequest = {
  pcm: Blob;
  startSeconds: number;
  endSeconds: number;
  sequence: number;
  speakerCount: number | null;
  /** End of the confirmable region; newest step after this stays provisional. */
  stableUntil: number;
};

export async function sendSpeakerWindow(
  liveSessionId: string,
  options: SpeakerWindowRequest,
): Promise<SpeakerWindowResult> {
  const form = new FormData();
  form.append("pcm", options.pcm, "window.pcm");
  form.append("start_seconds", options.startSeconds.toFixed(3));
  form.append("end_seconds", options.endSeconds.toFixed(3));
  form.append("sequence", String(options.sequence));
  form.append("stable_until", options.stableUntil.toFixed(3));
  if (options.speakerCount !== null) {
    form.append("speaker_count", String(options.speakerCount));
  }
  return request<SpeakerWindowResult>(
    `/api/v1/live-transcription/sessions/${liveSessionId}/speaker-window`,
    { method: "POST", body: form },
  );
}

export async function setLiveSpeakerAlias(
  liveSessionId: string,
  canonicalSpeaker: string,
  displayName: string,
): Promise<{ canonical_speaker: string; display_name: string }> {
  return request(
    `/api/v1/live-transcription/sessions/${liveSessionId}/speakers/${encodeURIComponent(canonicalSpeaker)}/alias`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
    },
  );
}

export async function clearLiveSpeakerAlias(
  liveSessionId: string,
  canonicalSpeaker: string,
): Promise<void> {
  const response = await fetch(
    `${getApiBaseUrl()}/api/v1/live-transcription/sessions/${liveSessionId}/speakers/${encodeURIComponent(canonicalSpeaker)}/alias`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
}

export type MeetingSpeakers = {
  meeting_id: string;
  speakers: string[];
  aliases: Record<string, string>;
};

export async function getMeetingSpeakers(meetingId: string): Promise<MeetingSpeakers> {
  return request<MeetingSpeakers>(`/api/v1/meetings/${meetingId}/speakers`);
}

export async function setMeetingSpeakerAlias(
  meetingId: string,
  canonicalSpeaker: string,
  displayName: string,
): Promise<{ canonical_speaker: string; display_name: string }> {
  return request(
    `/api/v1/meetings/${meetingId}/speakers/${encodeURIComponent(canonicalSpeaker)}/alias`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
    },
  );
}

export async function clearMeetingSpeakerAlias(
  meetingId: string,
  canonicalSpeaker: string,
): Promise<void> {
  const response = await fetch(
    `${getApiBaseUrl()}/api/v1/meetings/${meetingId}/speakers/${encodeURIComponent(canonicalSpeaker)}/alias`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    throw new ApiError(response.status, await readErrorMessage(response));
  }
}
