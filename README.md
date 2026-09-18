# Meeting Intelligence

A **web application** for local-first meeting transcription, speaker diarization and meeting
analysis.

The app runs entirely in a browser (macOS, Windows, Linux) and is served by a CPU-only backend.
No native desktop shell, no Apple-specific runtime dependency.

> **Status: audio capture only.** The browser recording → upload → ffmpeg conversion pipeline is
> implemented. Speech-to-text, diarization, LLM analysis and meeting history are **not implemented
> yet**.

## Architecture

### Browser

```text
Microphone
↓
MediaRecorder
↓
audio upload
```

Microphone access happens exclusively through standard browser APIs (`MediaDevices.getUserMedia`
and `MediaRecorder`). No native macOS/Windows APIs are involved.

### Backend

```text
FastAPI
↓
ffmpeg
↓
whisper.cpp
+
sherpa-onnx
↓
speaker-labelled transcript
↓
external LLM
```

The backend is designed to stay **CPU-compatible**: `whisper.cpp` and `sherpa-onnx` both run on
CPU, so the service can be developed on an Apple Silicon laptop and deployed later to a Linux
CPU server without code changes. To keep that promise reproducible, **the backend always runs
inside a Linux Docker container** (development included) — the host Python environment is never
the backend runtime.

The frontend intentionally runs **outside** Docker, natively on the host.

### Audio capture pipeline (implemented)

```text
Microphone
↓  getUserMedia (audio only, mono preferred)
MediaRecorder (64 kbps requested, 1 s chunks, in-memory)
↓  POST /api/recordings (multipart: audio, mime_type, client_duration_seconds)
FastAPI (streams the upload to disk, never buffers it fully)
↓  ffprobe validation → single ffmpeg process
meeting.mp3   (mono, ~96 kbps, playback/archive)
processing.wav (mono, 16 kHz, pcm_s16le, for STT + diarization)
```

- Recording format is negotiated with `MediaRecorder.isTypeSupported()`, preferring
  `audio/webm;codecs=opus`, then `audio/webm`, then `audio/mp4`; if none is supported the recorder
  is created without an explicit MIME type and the browser default is used (and reported).
- Microphone constraints are audio-only and conservative for speech models
  (`channelCount: {ideal: 1}`, echo cancellation / noise suppression / auto gain control preferred
  off). Unsupported optional constraints are ignored, never fatal.
- Uploads are validated by ffprobe (audio stream required); invalid or non-audio uploads are
  rejected. The client filename is never used for filesystem paths.
- Storage: `data/meetings/<recording_id>/` keeps `meeting.mp3` + `processing.wav`. The temporary
  source file is deleted on success, and the whole directory is removed on failure.
- The API returns measured metrics: input size, MP3 size, WAV size, duration, and ffmpeg
  conversion time in milliseconds.

Meeting audio and generated artifacts are stored on the local filesystem under `data/meetings/`
and are never committed to git.

## Stack

| Layer              | Technology                                                        |
| ------------------ | ----------------------------------------------------------------- |
| Frontend           | Next.js (App Router), TypeScript, Tailwind CSS — runs natively on the host |
| Browser audio      | `MediaDevices.getUserMedia`, `MediaRecorder`                      |
| Backend runtime    | Docker (`python:3.12-slim`), host architecture, Linux             |
| Backend            | Python 3.12, FastAPI, SQLAlchemy, SQLite (`aiosqlite`), uvicorn   |
| Audio processing   | ffmpeg (installed inside the backend container)                   |
| Speech-to-text     | whisper.cpp (CPU)                                                 |
| Speaker diarization| sherpa-onnx (CPU)                                                 |
| Meeting analysis   | External OpenAI-compatible LLM API (not implemented yet)          |
| Package management | uv (backend, inside Docker), npm (frontend, on host)              |
| Storage            | Local filesystem (`data/meetings/`), mounted into the container  |

## Browser and security requirements

- This is a web application; users only need a modern browser.
- Browsers only expose the microphone in a **secure context**. Production deployments must serve
  the app over **HTTPS**.
- `http://localhost` counts as a secure context, so microphone access works during local
  development without TLS.

## Platform support

- Backend: runs in a Linux container on every host, so macOS development and Linux CPU
  deployment share the same runtime (Python version, `sherpa-onnx` native libraries, ffmpeg).
- Frontend: any modern browser; no platform-specific code.
- Host Python is **not** the backend runtime, and host ffmpeg is **not** required to run the
  backend.
- Intentionally **not** used: MLX / mlx-whisper, Senko, CoreML, Metal-specific logic, CUDA,
  PyTorch, Electron, Tauri, native macOS or Windows APIs.

## Repository layout

```text
.
├── backend/          # FastAPI service (Python 3.12, uv) — run in Docker
│   ├── app/          # application code
│   ├── tests/        # pytest suite
│   └── Dockerfile    # Linux + Python 3.12 backend image
├── frontend/         # Next.js app (App Router, TypeScript, Tailwind) — run on host
│   └── src/
│       ├── app/      # routes
│       └── lib/      # browser capability helpers
├── compose.yaml      # backend development service (backend only)
├── data/
│   └── meetings/     # persistent local meeting storage (git-ignored, .gitkeep preserved)
└── README.md
```

## Development

Backend dependencies are **Docker-managed**. The container has its own environment inside the
image (`/opt/venv`); the host `.venv` is never used and never mounted.

### Backend (Docker only)

```bash
docker compose build backend
docker compose up -d backend
docker compose logs -f backend
```

- Backend URL: <http://localhost:8000>
- Health: <http://localhost:8000/health> → `{"status":"ok"}`
- Tests: `docker compose exec backend uv run --locked pytest`
- Lint: `docker compose exec backend uv run --locked ruff check .`
- Shell: `docker compose exec backend bash`

The source directory (`./backend`) is bind-mounted for reload, and `./data` is mounted so
meeting data survives container recreation. Inside the container the data directory is
`/data/meetings`.

### Frontend (native, not Dockerized)

```bash
cd frontend
npm install
npm run dev
```

- Frontend URL: <http://localhost:3000>
- Lint: `npm run lint`
- Type check: `npx tsc --noEmit`
- Build: `npm run build`

If port 3000 is already occupied by an unrelated local application, Next.js may pick another
port (for example 3001). That is expected; the default port is intentionally left as 3000.

### Meeting processing pipeline (backend)

```text
upload (POST /api/recordings)
        ↓  meeting row created (status: uploaded)
POST /api/v1/meetings/{id}/process   (status: queued)
        ↓  dedicated worker container (compose service `worker`)
claim (queued → processing, atomic)
        ↓
whisper.cpp large-v3-turbo Q8_0 (-ojf timestamps, no -nt)
        ↓
sherpa-onnx diarization (pyannote 3.0 + TitaNet Small, threshold 0.80,
or num_clusters = N when speaker_count is supplied)
        ↓
merge (max overlap → midpoint → 250 ms tolerance → unresolved stays unresolved)
        ↓
transcript turns (Kişi N session-local labels)   (status: completed)
```

- The API never runs inference; a separate worker process claims jobs from the shared
  SQLite database (`data/app.db`, git-ignored). No Redis/Celery.
- `GET /api/v1/meetings/{id}` → status; `GET /api/v1/meetings/{id}/transcript` → turns.
- Unresolved words are preserved as their own turns labelled `Bilinmeyen` (never
  attributed to a neighbouring speaker). No voiceprints or embeddings are stored.
- Models are mounted, never baked into image layers
  (`/models/whisper`, `/models/diarization`); paths and thread counts are configured
  with `MEETING_*` environment variables (see `backend/app/config.py`).

```bash
docker compose up -d backend worker     # API + processing worker
docker compose logs -f worker
docker compose exec backend uv run --locked pytest -m integration -q   # real-model E2E
```

### Transcription providers (local default, ElevenLabs opt-in)

```bash
MEETING_TRANSCRIPTION_PROVIDER=local        # default: whisper.cpp + TitaNet, audio stays local
MEETING_TRANSCRIPTION_PROVIDER=elevenlabs   # opt-in cloud path: ElevenLabs Scribe v2
ELEVENLABS_API_KEY=...                      # only required for the elevenlabs provider
ELEVENLABS_STT_MODEL=scribe_v2
ELEVENLABS_LANGUAGE_CODE=tur
ELEVENLABS_TIMEOUT_SECONDS=120
ELEVENLABS_DIARIZATION_THRESHOLD=           # optional; only sent without a known speaker count
```

- **Local (default):** whisper.cpp v1.9.4 large-v3-turbo Q8_0 + TitaNet diarization
  (threshold 0.80, `min_duration_on 0.3` / `min_duration_off 0.5`) and the production merge.
  Audio never leaves the machine. Behavior is unchanged by this feature.
- **ElevenLabs (opt-in):** audio is uploaded to ElevenLabs, internet and quota/provider
  availability are required, and **normal ElevenLabs retention may apply** for the current
  non-zero-retention account. Word timestamps and `speaker_id` come directly from the
  provider, so the whisper↔diarization merge is not used on this path. A requested
  `speaker_count` is forwarded as `num_speakers` (documented as a maximum expected
  speaker count); `diarization_threshold` is only sent when no speaker count is given.
- Failures are explicit: the meeting is marked `failed` with a safe message and there is
  **no silent fallback** to the local pipeline (that would make latency, quality and
  privacy behavior unpredictable).
- The provider and model are stored on the meeting (`transcription_provider`,
  `transcription_model`) and exposed on `GET /api/v1/meetings/{id}`; keys, headers and raw
  provider responses are never persisted or exposed.
- Real cloud verification is opt-in only:
  `RUN_ELEVENLABS_INTEGRATION=1 docker compose run --rm -e ELEVENLABS_API_KEY ... pytest -m integration`.

### Grounded meeting analysis (backend)

```text
transcript completed
        ↓
POST /api/v1/meetings/{id}/analyze   (status: queued; 409 until the transcript is done)
        ↓  worker priority: transcription jobs first, then analysis jobs
provider (OpenAI-compatible chat completions, JSON-only response)
        ↓  local Pydantic + semantic validation (ordinals must exist, owners must be
           existing "Kişi N" or null, "Bilinmeyen" can never own an action)
        ↓  one repair attempt if the first response is invalid, then failed
meeting_analyses row (summary, topics, decisions, action items, important moments)
        ↓
GET /api/v1/meetings/{id}/analysis   (timestamps derived from persisted turns)
```

- The model only references transcript **ordinals**; the backend resolves them to
  `timestamp_seconds`, so provider timestamps are never trusted.
- LLM failures never touch the transcript, the meeting status or the audio artifacts;
  analysis is an independent one-to-one row with its own status.
- `MEETING_LLM_PROVIDER=mock` selects a deterministic local provider for tests and the
  local E2E; the default is `openai_compatible`.
- Secrets are read from `MEETING_LLM_BASE_URL`, `MEETING_LLM_API_KEY`,
  `MEETING_LLM_MODEL` (never logged, never returned); `.env` files are git-ignored.
- **Eligibility guard (temporary):** endpoints reserved for coding-agent traffic
  (currently `https://opencode.ai/zen/go/v1`) are refused before any meeting content is
  sent — `Configured LLM endpoint is restricted to coding-agent traffic and is not
  enabled for meeting analysis.`
- Long transcripts are rejected with a clear error instead of silent truncation
  (`MEETING_LLM_MAX_TRANSCRIPT_CHARS`).

### Frontend meeting workflow

- `/` — record (MediaRecorder, unchanged MIME negotiation), optional speaker count
  (`Otomatik` → `speaker_count=null`), upload, queue transcription, status polling
  (one timer at a time, stops on completed/failed) and the meeting history.
- `/meetings/[id]` — reconstructs everything from the backend on refresh: status,
  MP3 player (native `<audio>`, served by `GET /api/v1/meetings/{id}/audio` with
  Range support), speaker-attributed transcript and the analysis sections.
- Clicking any timestamp (transcript turn, decision, action item, important moment)
  seeks the audio player to `timestamp_seconds` and starts playback.
- Analysis: `Toplantıyı Analiz Et` → `POST /analyze` → polling → sections
  (Özet / Konular / Kararlar / Aksiyonlar / Önemli Anlar). If no eligible LLM provider
  is configured the backend returns 503 and the UI shows a non-destructive message
  while the transcript stays fully usable.
- Frontend tests (vitest + Testing Library): `cd frontend && npm test`.

### Frontend API base URL

Copy `frontend/.env.local.example` to `frontend/.env.local` and adjust if needed:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Development CORS origins are configured centrally in the backend
(`MEETING_CORS_ORIGINS`, JSON list; defaults cover `localhost`/`127.0.0.1` on ports 3000 and 3100).
Never use `allow_origins=["*"]`.

## Known issues

- **sherpa-onnx is pinned to `1.10.46`.** The pin is validated inside the Linux backend
  container, not on the host. `1.13.8` fails there with
  `ImportError: libonnxruntime.so: cannot open shared object file` — its manylinux wheels do not
  bundle the ONNX Runtime shared library. `1.10.46` is the newest release whose **Linux** wheels
  are self-contained for both `x86_64` and `aarch64`, and it already exposes the diarization API
  (`sherpa_onnx.OfflineSpeakerDiarization`). The pin should be revisited once upstream publishes
  self-contained manylinux wheels.

## License

Not specified yet.
