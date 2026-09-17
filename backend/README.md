# Meeting Intelligence — Backend

FastAPI service for the Meeting Intelligence web application.

- Python 3.12, managed with [uv](https://docs.astral.sh/uv/)
- CPU-only, cross-platform (macOS / Linux)
- No Apple-specific runtime dependencies

## Development

```bash
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check .
```

`GET /health` returns `{"status": "ok"}`.

See the repository root `README.md` for the full architecture description.
