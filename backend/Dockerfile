# Backend development runtime: Linux + Python 3.12.
# The base image is multi-arch, so the build uses the host architecture
# automatically (Apple Silicon locally, x86_64 on future Linux deployments).

# Production STT runtime: whisper.cpp pinned to the validated release. CPU only,
# static libs, same build flags as the benchmark round. The model itself is NOT
# baked into the image (multi-GB assets stay in the mounted model cache).
FROM python:3.12-slim AS whisper-cpp-build

ARG WHISPER_CPP_REF=v1.9.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" https://github.com/ggml-org/whisper.cpp.git /opt/whisper.cpp

RUN cmake \
        -S /opt/whisper.cpp \
        -B /opt/whisper.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_SERVER=OFF \
    && cmake --build /opt/whisper.cpp/build --config Release -j"$(nproc)"

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
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=whisper-cpp-build /opt/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli

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
