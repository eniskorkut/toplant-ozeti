# Isolated scoring environment for the diarization benchmark.
# dscore is pinned to an exact commit; nothing here touches the application.
FROM python:3.12-slim

ARG DSCORE_SHA=e02f949ac6592279300a2c33d03daf9e0c12fd27

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/nryant/dscore.git /opt/dscore \
    && cd /opt/dscore \
    && git checkout "${DSCORE_SHA}" \
    && pip install --no-cache-dir -r requirements.txt

WORKDIR /opt/dscore
