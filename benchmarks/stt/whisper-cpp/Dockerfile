# Benchmark image: whisper.cpp CLI, CPU only, built from a pinned release tag.
# Isolated from the application runtime.
#
# CPU-only on purpose: no CUDA, Metal, CoreML, ROCm or Vulkan is enabled.
# GGML_NATIVE is left at its default (ON) so the build uses the portable
# ggml CPU kernels for the build platform (Linux aarch64 here, x86_64 builds
# work the same way).
FROM python:3.12-slim AS build

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
    && cmake --build /opt/whisper.cpp/build --config Release -j"$(nproc)" \
    && printf '%s\n' "${WHISPER_CPP_REF}" > /opt/whisper.cpp/VERSION


FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        time \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=build /opt/whisper.cpp/models/download-ggml-model.sh /opt/whisper.cpp/models/download-ggml-model.sh
COPY --from=build /opt/whisper.cpp/VERSION /opt/whisper.cpp/VERSION

ENV OMP_NUM_THREADS=8

WORKDIR /bench
COPY transcribe.py /bench/transcribe.py
