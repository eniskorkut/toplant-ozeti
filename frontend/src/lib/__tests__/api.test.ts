import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  deleteMeeting,
  getMeeting,
  getTranscript,
  getTranscriptionProviders,
  listMeetings,
  processMeeting,
  uploadRecording,
} from "@/lib/api";

const fetchMock = vi.fn();

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  fetchMock.mockReset();
  vi.unstubAllGlobals();
});

function stubFetch() {
  vi.stubGlobal("fetch", fetchMock);
}

describe("api client", () => {
  it("parses a meeting status response", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(
      jsonResponse({
        meeting_id: "m1",
        status: "completed",
        created_at: "2026-09-18T10:00:00",
        duration_seconds: 12.5,
        requested_speaker_count: null,
        processing_error: null,
        has_transcript: true,
      }),
    );

    const meeting = await getMeeting("m1");

    expect(meeting.status).toBe("completed");
    expect(meeting.has_transcript).toBe(true);
    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/api/v1/meetings/m1");
  });

  it("parses a transcript response without altering speaker labels", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(
      jsonResponse({
        meeting_id: "m1",
        status: "completed",
        duration_seconds: 10,
        speakers: ["Kişi 1"],
        unresolved_label: "Bilinmeyen",
        unresolved_turns: 1,
        turns: [
          { ordinal: 0, speaker: "Kişi 1", start_seconds: 0, end_seconds: 1, text: "a" },
          { ordinal: 1, speaker: "Bilinmeyen", start_seconds: 1, end_seconds: 2, text: "b" },
        ],
      }),
    );

    const transcript = await getTranscript("m1");

    expect(transcript.turns.map((turn) => turn.speaker)).toEqual(["Kişi 1", "Bilinmeyen"]);
    expect(transcript.unresolved_turns).toBe(1);
  });

  it("raises a typed ApiError with the backend detail", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(jsonResponse({ detail: "Meeting not found" }, 404));

    await expect(getMeeting("missing")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "Meeting not found",
    });
  });

  it("posts the local provider and speaker count when queueing processing", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(jsonResponse({ meeting_id: "m1", status: "queued" }));

    await processMeeting("m1", { speakerCount: 3, transcriptionProvider: "local" });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/v1/meetings/m1/process");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      speaker_count: 3,
      transcription_provider: "local",
    });
  });

  it("posts the elevenlabs provider", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(jsonResponse({ meeting_id: "m1", status: "queued" }));

    await processMeeting("m1", { speakerCount: null, transcriptionProvider: "elevenlabs" });

    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({
      speaker_count: null,
      transcription_provider: "elevenlabs",
    });
  });

  it("fetches provider capabilities", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(
      jsonResponse({
        default: "local",
        providers: [
          { id: "local", available: true, cloud: false, label: "Yerel" },
          { id: "elevenlabs", available: false, cloud: true, label: "ElevenLabs" },
        ],
      }),
    );

    const capabilities = await getTranscriptionProviders();

    expect(fetchMock.mock.calls[0][0]).toBe(
      "http://localhost:8000/api/v1/transcription/providers",
    );
    expect(capabilities.default).toBe("local");
    expect(capabilities.providers[1].available).toBe(false);
  });

  it("lists meetings through the versioned endpoint", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(jsonResponse({ meetings: [], count: 0 }));

    await listMeetings();

    expect(fetchMock.mock.calls[0][0]).toBe("http://localhost:8000/api/v1/meetings");
  });

  it("uploads a recording as multipart form data", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(jsonResponse({ meeting_id: "m1", recording_id: "m1" }));

    await uploadRecording(new Blob(["x"], { type: "audio/webm" }), {
      mimeType: "audio/webm",
      durationSeconds: 2,
      filename: "recording.webm",
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/recordings");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("mime_type")).toBe("audio/webm");
  });

  it("deletes a meeting through the versioned endpoint", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(new Response(null, { status: 204 }));

    await expect(deleteMeeting("m1")).resolves.toBeUndefined();

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:8000/api/v1/meetings/m1");
    expect(init.method).toBe("DELETE");
  });

  it("maps a 409 delete conflict to an ApiError", async () => {
    stubFetch();
    fetchMock.mockResolvedValue(
      jsonResponse({ detail: "Meeting is being processed and cannot be deleted." }, 409),
    );

    await expect(deleteMeeting("m1")).rejects.toMatchObject({
      status: 409,
      message: "Meeting is being processed and cannot be deleted.",
    });
  });

  it("ApiError keeps the status code for callers", () => {
    const error = new ApiError(503, "not configured");
    expect(error.status).toBe(503);
    expect(error).toBeInstanceOf(Error);
  });
});
