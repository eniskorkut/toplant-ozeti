#!/usr/bin/env python3
"""Real ElevenLabs live-path E2E against the running stack (backend container).

Replays the known 22 s two-person sample through the REAL application path:

  realtime single-use token (our API) -> Scribe v2 Realtime WebSocket
  rolling windows (our API) -> Scribe v2 batch diarization + canonical mapping
  archival upload (our API, live_session_id) -> alias migration
  final processing (our API, provider=elevenlabs) -> authoritative transcript

It prints only safe aggregates (delays, counts, statuses) and never transcript
text, tokens or provider payloads. Run inside the backend container:

    docker compose exec -T backend python /tmp/bench/run_live_e2e.py
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import httpx
from websockets.sync.client import connect as ws_connect

API = "http://localhost:8000"
AUDIO = Path("/data/meetings/8266cc18930b40e6a3740c48bc543705/processing.wav")
REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
LOOKBACK_SECONDS = 12.0
STEP_SECONDS = 4.0
FIRST_ATTEMPT_SECONDS = 8.0
# Browser cadence: PcmCapture emits ~100 ms chunks (1600 samples at 16 kHz).
CHUNK_SECONDS = 0.1
PARTIAL_MIN_GAP_SECONDS = 1.0


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getframerate() != 16_000 or wav_file.getnchannels() != 1:
            raise SystemExit("expected 16 kHz mono wav")
        return wav_file.readframes(wav_file.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="Test Kullanıcı")
    parser.add_argument("--time-compression", type=float, default=1.0)
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    pcm = read_pcm(AUDIO)
    duration = len(pcm) / 32_000

    with httpx.Client(base_url=API, timeout=60.0) as client:
        health = client.get("/health")
        print("health:", health.status_code)

        session = client.post("/api/v1/live-transcription/sessions").json()
        live_session_id = session["live_session_id"]
        print("live_session_id_present:", bool(live_session_id))

        token_response = client.post(
            "/api/v1/transcription/providers/elevenlabs/realtime-token"
        )
        if token_response.status_code != 200:
            print("realtime_token_status:", token_response.status_code)
            return 1
        token = token_response.json()["token"]
        print("realtime_token_received: true (never printed)")

        url = (
            f"{REALTIME_URL}?model_id=scribe_v2_realtime&language_code=tur"
            f"&include_timestamps=true&commit_strategy=vad"
            f"&vad_silence_threshold_secs=1.5&vad_threshold=0.4"
            f"&min_speech_duration_ms=100&min_silence_duration_ms=100"
            f"&token={token}"
        )

        started = time.perf_counter()
        partial_delays: list[float] = []
        partial_events: list[float] = []  # wall-clock elapsed per partial event
        commits_info: list[dict] = []  # wall + word anchors per committed segment
        committed_delays: list[float] = []
        parts = 0
        commits = 0
        words_seen = 0
        last_word_end = 0.0
        awaiting_onset = True
        label_delays: list[float] = []
        rolling_requests = 0
        rolling_seconds = 0.0
        sequence = 1
        next_window_start = FIRST_ATTEMPT_SECONDS
        step = STEP_SECONDS

        with ws_connect(url, open_timeout=15, close_timeout=5) as socket:
            for chunk_index in range(int(duration / CHUNK_SECONDS) + 1):
                start_sample = int(chunk_index * CHUNK_SECONDS * 16_000)
                end_sample = start_sample + int(CHUNK_SECONDS * 16_000)
                chunk = pcm[start_sample * 2 : end_sample * 2]
                if chunk:
                    socket.send(
                        json.dumps(
                            {
                                "message_type": "input_audio_chunk",
                                "audio_base_64": base64.b64encode(chunk).decode(),
                                "sample_rate": 16_000,
                            }
                        )
                    )

                # Collect whatever arrived without blocking the pacing.
                try:
                    while True:
                        message = json.loads(socket.recv(timeout=0.01))
                        kind = message.get("message_type", "")
                        elapsed = time.perf_counter() - started
                        if kind == "partial_transcript" and message.get("text"):
                            parts += 1
                            partial_events.append(elapsed)
                            # Legacy anchor (previous committed speech end): kept for
                            # before/after comparison with the earlier metric.
                            if awaiting_onset and last_word_end > 0:
                                delay = elapsed - last_word_end
                                if 0 <= delay <= 5.0:
                                    partial_delays.append(delay)
                                awaiting_onset = False
                        elif kind.startswith("committed_transcript"):
                            commits += 1
                            words = message.get("words") or []
                            if words:
                                own_start = float(words[0].get("start", 0.0))
                                own_end = float(words[-1].get("end", 0.0))
                                committed_delays.append(max(0.0, elapsed - own_end))
                                commits_info.append(
                                    {
                                        "wall": elapsed,
                                        "word_start": own_start,
                                        "word_end": own_end,
                                    }
                                )
                                words_seen += len(words)
                                last_word_end = max(last_word_end, own_end)
                            awaiting_onset = True
                        elif kind in ("error", "auth_error"):
                            print("realtime_error:", kind)
                except TimeoutError:
                    pass

                # Rolling diarization snapshots (long-context lookback).
                current_end = (chunk_index + 1) * CHUNK_SECONDS
                while next_window_start <= current_end:
                    snapshot_end = next_window_start
                    window_start = max(0.0, snapshot_end - LOOKBACK_SECONDS)
                    window_end = snapshot_end
                    window_pcm = pcm[
                        int(window_start * 32_000) : int(window_end * 32_000)
                    ]
                    stable_until = max(window_start, window_end - STEP_SECONDS)
                    response = client.post(
                        f"/api/v1/live-transcription/sessions/{live_session_id}/speaker-window",
                        files={"pcm": ("window.pcm", window_pcm, "application/octet-stream")},
                        data={
                            "start_seconds": f"{window_start:.3f}",
                            "end_seconds": f"{window_end:.3f}",
                            "sequence": str(sequence),
                            "stable_until": f"{stable_until:.3f}",
                        },
                    )
                    if response.status_code == 200:
                        body = response.json()
                        responded_at = time.perf_counter() - started
                        for assignment in body.get("assignments", []):
                            if assignment.get("provisional"):
                                continue
                            label_delays.append(
                                max(0.0, responded_at - float(assignment["end"]))
                            )
                        rolling_requests += 1
                        rolling_seconds += window_end - window_start
                        sequence += 1
                    else:
                        print("rolling_window_status:", response.status_code)
                    next_window_start += step

                if chunk_index < int(duration / CHUNK_SECONDS):
                    target = started + (chunk_index + 1) * CHUNK_SECONDS * args.time_compression
                    sleep_for = target - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            # Flush: ask the server to commit the tail, then drain briefly.
            socket.send(json.dumps({"message_type": "commit"}))
            drain_until = time.perf_counter() + 3.0
            while time.perf_counter() < drain_until:
                try:
                    message = json.loads(socket.recv(timeout=0.2))
                    kind = message.get("message_type", "")
                    if kind.startswith("committed_transcript"):
                        commits += 1
                        words = message.get("words") or []
                        if words:
                            commits_info.append(
                                {
                                    "wall": time.perf_counter() - started,
                                    "word_start": float(words[0].get("start", 0.0)),
                                    "word_end": float(words[-1].get("end", 0.0)),
                                }
                            )
                    elif kind == "partial_transcript" and message.get("text"):
                        parts += 1
                        partial_events.append(time.perf_counter() - started)
                except TimeoutError:
                    continue

        # Fair partial metric: for each committed segment, the first partial event
        # received after the previous commit is anchored at this segment's first
        # word start (speech onset), not at the previous silence.
        onset_delays: list[float] = []
        for index, info in enumerate(commits_info):
            window_start = commits_info[index - 1]["wall"] if index > 0 else 0.0
            candidates = [
                event
                for event in partial_events
                if window_start <= event <= info["wall"]
            ]
            if not candidates:
                continue
            onset_wall = info["word_start"]  # stream wall ~= audio timeline
            delay = candidates[0] - onset_wall
            if 0 <= delay <= 10.0:
                onset_delays.append(delay)

        session_state = client.get(
            f"/api/v1/live-transcription/sessions/{live_session_id}"
        ).json()
        speakers = session_state["speakers"]
        print("live_speakers:", [s["canonical_speaker"] for s in speakers])
        print("live_windows_received:", session_state["windows_received"])
        print("live_label_switches:", session_state["label_switches"])

        # Rename while the session is still live.
        if speakers:
            canonical = speakers[0]["canonical_speaker"]
            rename = client.put(
                f"/api/v1/live-transcription/sessions/{live_session_id}/speakers/"
                f"{canonical}/alias",
                json={"display_name": args.alias},
            )
            print("live_alias_status:", rename.status_code)
            print(
                "live_alias_visible:",
                client.get(f"/api/v1/live-transcription/sessions/{live_session_id}")
                .json()["aliases"]
                .get(canonical)
                == args.alias,
            )

        if args.skip_upload or args.skip_finalize:
            print("e2e_measure_only: true")
            print("e2e_report:", json.dumps({
                "recording_seconds": round(duration, 2),
                "realtime_partial_events": parts,
                "realtime_committed_events": commits,
                "realtime_words_seen": words_seen,
                "first_partial_at_s": round(partial_events[0], 3) if partial_events else None,
                "first_partial_from_speech_onset_median_s": round(
                    statistics.median(onset_delays), 3
                )
                if onset_delays
                else None,
                "first_partial_delay_median_s": round(statistics.median(partial_delays), 3)
                if partial_delays
                else None,
                "committed_text_delay_median_s": round(
                    statistics.median(committed_delays), 3
                )
                if committed_delays
                else None,
                "speaker_label_delay_median_s": round(statistics.median(label_delays), 3)
                if label_delays
                else None,
                "rolling_requests": rolling_requests,
                "rolling_audio_seconds": round(rolling_seconds, 1),
                "final_audio_seconds": 0.0,
            }))
            return 0

        # Upload the archival recording as webm and migrate the live state.
        import subprocess

        webm_path = Path("/tmp/bench/e2e-archival.webm")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                "/dev/stdin",
                "-c:a",
                "libopus",
                "-b:a",
                "64k",
                str(webm_path),
            ],
            input=pcm,
            check=True,
        )
        upload = client.post(
            "/api/recordings",
            files={"audio": ("recording.webm", webm_path.read_bytes(), "audio/webm;codecs=opus")},
            data={
                "mime_type": "audio/webm;codecs=opus",
                "client_duration_seconds": f"{duration:.3f}",
                "live_session_id": live_session_id,
            },
            timeout=300.0,
        )
        print("upload_status:", upload.status_code)
        meeting_id = upload.json()["meeting_id"]
        print("meeting_id:", meeting_id)

        migrated = client.get(f"/api/v1/meetings/{meeting_id}/speakers").json()
        print("meeting_aliases_after_upload:", migrated["aliases"])
        print("live_session_released:", (
            client.get(f"/api/v1/live-transcription/sessions/{live_session_id}").status_code
            == 404
        ))

        # Final authoritative Scribe v2 pass through the worker.
        process = client.post(
            f"/api/v1/meetings/{meeting_id}/process",
            json={"speaker_count": None, "transcription_provider": "elevenlabs"},
        )
        print("process_status:", process.status_code)
        deadline = time.time() + 180
        status = "queued"
        while time.time() < deadline:
            status = client.get(f"/api/v1/meetings/{meeting_id}").json()["status"]
            if status in ("completed", "failed"):
                break
            time.sleep(2)
        print("final_status:", status)

        if status == "completed":
            transcript = client.get(f"/api/v1/meetings/{meeting_id}/transcript").json()
            print("final_turns:", len(transcript["turns"]))
            print("final_speakers:", transcript["speakers"])
            refreshed = client.get(f"/api/v1/meetings/{meeting_id}/speakers").json()
            print("alias_after_refresh:", refreshed["aliases"])

        # New-meeting isolation: an unrelated meeting must not carry the alias.
        listing = client.get("/api/v1/meetings").json()["meetings"]
        other = next((m for m in listing if m["meeting_id"] != meeting_id), None)
        if other:
            other_aliases = client.get(
                f"/api/v1/meetings/{other['meeting_id']}/speakers"
            ).json()["aliases"]
            print("other_meeting_has_alias:", bool(other_aliases))

    def med(values: list[float]) -> float | None:
        return round(statistics.median(values), 3) if values else None

    print("e2e_report:", json.dumps({
        "recording_seconds": round(duration, 2),
        "realtime_partial_events": parts,
        "realtime_committed_events": commits,
        "realtime_words_seen": words_seen,
        "first_partial_at_s": round(partial_events[0], 3) if partial_events else None,
        "first_partial_from_speech_onset_median_s": med(onset_delays),
        "first_partial_delay_median_s": med(partial_delays),
        "committed_text_delay_median_s": med(committed_delays),
        "speaker_label_delay_median_s": med(label_delays),
        "rolling_requests": rolling_requests,
        "rolling_audio_seconds": round(rolling_seconds, 1),
        "final_audio_seconds": round(duration, 2),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
