# Benchmark image: faster-whisper on CPU (CTranslate2, int8).
# Isolated from the application runtime: nothing here is installed in the
# production backend image or declared in backend/pyproject.toml.
FROM python:3.12-slim

# `time` provides /usr/bin/time -v for peak RSS measurement.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        time \
    && rm -rf /var/lib/apt/lists/*

# Pinned on purpose: the benchmark must be reproducible.
RUN pip install --no-cache-dir faster-whisper==1.2.1

ENV HF_HOME=/cache/huggingface \
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /bench
COPY transcribe.py /bench/transcribe.py
