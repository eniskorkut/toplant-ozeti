# Backend development runtime: Linux + Python 3.12.
# The base image is multi-arch, so the build uses the host architecture
# automatically (Apple Silicon locally, x86_64 on future Linux deployments).
FROM python:3.12-slim

# The installer is not needed: uv is copied from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

# Keep the project environment outside the bind-mounted source tree so the
# host .venv can never shadow the container environment.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime system packages for the current baseline only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (better layer caching). uv.lock is authoritative:
# --locked fails the build if pyproject.toml and uv.lock disagree, and never
# regenerates the lock file.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

EXPOSE 8000

CMD ["uv", "run", "--locked", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
