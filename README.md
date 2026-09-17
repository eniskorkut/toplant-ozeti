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
CPU server without code changes.

Meeting audio and generated artifacts are stored on the local filesystem under `data/meetings/`
and are never committed to git.

## Stack

| Layer              | Technology                                                        |
| ------------------ | ----------------------------------------------------------------- |
| Frontend           | Next.js (App Router), TypeScript, Tailwind CSS                    |
| Browser audio      | `MediaDevices.getUserMedia`, `MediaRecorder`                      |
| Backend            | Python 3.12, FastAPI, SQLAlchemy, SQLite (`aiosqlite`), uvicorn   |
| Audio processing   | ffmpeg                                                            |
| Speech-to-text     | whisper.cpp (CPU)                                                 |
| Speaker diarization| sherpa-onnx (CPU)                                                 |
| Meeting analysis   | External OpenAI-compatible LLM API (not implemented yet)          |
| Package management | uv (backend), npm (frontend)                                      |
| Storage            | Local filesystem (`data/meetings/`)                               |

## Browser and security requirements

- This is a web application; users only need a modern browser.
- Browsers only expose the microphone in a **secure context**. Production deployments must serve
  the app over **HTTPS**.
- `http://localhost` counts as a secure context, so microphone access works during local
  development without TLS.

## Platform support

- Backend: macOS and Linux (CPU only).
- Intentionally **not** used: MLX / mlx-whisper, Senko, CoreML, Metal-specific logic, CUDA,
  PyTorch, Electron, Tauri, native macOS or Windows APIs.

## Repository layout

```text
.
├── backend/          # FastAPI service (Python 3.12, uv)
│   ├── app/          # application code
│   └── tests/        # pytest suite
├── frontend/         # Next.js app (App Router, TypeScript, Tailwind)
│   └── src/
│       ├── app/      # routes
│       └── lib/      # browser capability helpers
├── data/
│   └── meetings/     # local meeting storage (git-ignored, .gitkeep preserved)
└── README.md
```

## Development

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload   # http://127.0.0.1:8000
uv run pytest
uv run ruff check .
```

`GET /health` → `{"status": "ok"}`

### Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:3000
npm run lint
npm run build
```

### System dependency: ffmpeg

ffmpeg must be available on the system `PATH` (the code never hard-codes platform paths).

- macOS: `brew install ffmpeg`
- Debian/Ubuntu: `sudo apt install ffmpeg`

Backend code can report its availability via `app.services.ffmpeg` (`is_ffmpeg_available()`,
`get_ffmpeg_info()`).

## Known issues

- **sherpa-onnx is pinned to `1.10.46`.** The currently published macOS arm64 wheels
  (`sherpa-onnx >= 1.12`) do not bundle `libonnxruntime.dylib`, so importing the package fails
  with `Library not loaded: @rpath/libonnxruntime.dylib`. `1.10.46` is the newest release whose
  macOS and manylinux wheels are self-contained, keeps development and Linux deployment on the
  same version, and already exposes the diarization API. Remove the pin once the upstream wheel
  fix is released.

## License

Not specified yet.
