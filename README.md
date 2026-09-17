# Meeting Intelligence

A **web application** for local-first meeting transcription, speaker diarization and meeting
analysis.

The app runs entirely in a browser (macOS, Windows, Linux) and is served by a CPU-only backend.
No native desktop shell, no Apple-specific runtime dependency.

> **Status: scaffolding only.** Recording, transcription, diarization and LLM analysis are
> **not implemented yet**. This repository currently contains the project skeleton, the
> `/health` endpoint, browser capability detection helpers and an ffmpeg availability check.

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
