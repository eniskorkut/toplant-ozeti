# Meeting Intelligence — Backend

FastAPI service for the Meeting Intelligence web application.

- Python 3.12 in a Linux Docker container, managed with [uv](https://docs.astral.sh/uv/)
- CPU-only, cross-platform (macOS / Linux hosts, Linux runtime)
- No Apple-specific runtime dependencies
- **The host Python environment is not the backend runtime.** Dependencies are installed inside
  the image at `/opt/venv` from `pyproject.toml` + `uv.lock` (`uv sync --locked`).

## Development

```bash
docker compose build backend      # from the repository root
docker compose up -d backend
docker compose logs -f backend
docker compose exec backend bash
docker compose exec backend uv run --locked pytest
docker compose exec backend uv run --locked ruff check .
```

The container already runs `uvicorn --reload` against the bind-mounted source.

`GET /health` returns `{"status": "ok"}`.

See the repository root `README.md` for the full architecture description.
