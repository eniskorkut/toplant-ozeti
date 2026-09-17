# STT benchmark environment (temporary, isolated)

Purpose: measure serious CPU speech-to-text candidates on the same real Turkish
recordings before choosing a production implementation.

**Nothing here is part of the application runtime.**

- No STT dependency is added to `backend/pyproject.toml`.
- No STT build tool is added to the production backend image.
- Both candidates run in their own Docker images, CPU only, Linux.
- No MLX / Metal / CoreML / CUDA / ROCm / Vulkan.

## Matrix (4 configurations × 2 samples)

| Configuration | Engine | Model | Decoding |
|---|---|---|---|
| A1 `fw-small-int8-b1` | faster-whisper 1.2.1 (CTranslate2 CPU int8) | `small` multilingual | sequential, `beam_size=5` |
| A2 `fw-small-int8-b8` | faster-whisper 1.2.1 (CTranslate2 CPU int8) | `small` multilingual | official `BatchedInferencePipeline`, `batch_size=8`, fixed 30 s clips |
| B1 `wc-small-q5_1` | whisper.cpp v1.9.4 (`whisper-cli`) | `small-q5_1` (official) | `-bs 5` |
| B2 `wc-small-q8_0` | whisper.cpp v1.9.4 (`whisper-cli`) | `small-q8_0` (quantized locally, official source) | `-bs 5` |

Samples (same reference text, different microphone distance):

| Sample | Recording | Condition |
|---|---|---|
| `sample_far` | `fcdf1737901d41ddacd14429d39a4183` | quiet/distant |
| `sample_near` | newest other recording | controlled close take |

`sample_far` is pinned by id; `sample_near` is the newest `data/meetings/*/processing.wav`
that is not the pinned far recording. Override with `--far-audio` / `--near-audio`.

## Fairness rules implemented

- Both engines read the **same** `processing.wav` (mounted read-only at `/audio`).
- Same language (`tr`), thread count (8), beam size (5), VAD off.
- Peak RSS from GNU `/usr/bin/time -v` inside the container.
- No image build, model download, quantization or warmup time is measured.
- 1 warmup run (excluded) + 3 measured runs per configuration and sample.
- Model caches live in `benchmarks/stt/.cache/` (git-ignored).

## Tiers

Two tiers share the same samples, reference text, fairness rules and scoring:

- `--tier small` (default): A1/A2 faster-whisper `small` int8, B1 whisper.cpp `small-q5_1`,
  B2 whisper.cpp `small-q8_0` (locally quantized from the official f16 model).
- `--tier turbo`: T1 faster-whisper `large-v3-turbo` int8 (sequential only — the previous
  round showed batching worsens WER), T2 whisper.cpp `large-v3-turbo-q8_0` (official model,
  not locally quantized).

```bash
python3 benchmarks/stt/run_benchmark.py --setup                  # small tier
python3 benchmarks/stt/run_benchmark.py --skip-setup
python3 benchmarks/stt/run_benchmark.py --tier turbo --setup     # downloads + verifies official turbo models
python3 benchmarks/stt/run_benchmark.py --tier turbo --skip-setup
```

Turbo results are written to `results/turbo-matrix.{json,md}` and `results/turbo-runs/`
so the small-tier artifacts are never overwritten. Model verification data
(expected vs. local SHA256 and size) is written to
`results/model-metadata-turbo.json` (hash/size information only, no speech content).

## Model verification (turbo tier)

- **whisper.cpp:** official `ggerganov/whisper.cpp` distribution, `ggml-large-v3-turbo-q8_0.bin`.
  Expected size and SHA256 are read from the official Hugging Face repository and compared with
  the local file before benchmarking. The model is **not** locally quantized.
- **faster-whisper:** the repo id is resolved from `faster_whisper.utils._MODELS`
  (`large-v3-turbo` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, which Hugging Face
  currently redirects upstream). `model.bin` is verified against the official SHA256/size.

## Evaluation bands (project thresholds, applied after the fact, never adjusted)

WER: `<= 0.10` excellent · `<= 0.15` strong · `<= 0.20` possibly usable, review errors ·
`> 0.20` not acceptable as final production quality.

RTF: `<= 0.15` excellent CPU speed · `<= 0.30` good for post-meeting processing ·
`<= 0.50` acceptable but slower · `> 0.50` too slow for the preferred UX.

## Resource safety

The Docker VM on the development machine has ~8.1 GB RAM and 1 GB swap. Peak RSS per
configuration is captured with `time -v`, swap activity is checked before/after each
configuration, and a candidate is stopped instead of pushing the VM into memory pressure.

## Documented behavior differences (not hidden)

- **A2 batched mode:** with `vad_filter=False`, the official API requires explicit
  clip boundaries for audio longer than 30 s (it raises `RuntimeError` otherwise).
  Fixed 30 s windows are passed explicitly — the same window length the pipeline
  uses internally when VAD is enabled. Each clip is decoded independently, so
  cross-window context (`condition_on_previous_text`) is lost in batched mode;
  this is inherent to the official API.
- **B1/B2:** `whisper-cli` reports its own `total time` (encode + decode), which is
  used as the inference time; the sequential faster-whisper configs use a
  `perf_counter` measurement around the decode call. Both exclude process startup.
- **Q8_0 model:** `small` Q8_0 is not published by whisper.cpp, so it is generated
  locally from the officially distributed `ggml-small.bin` using whisper.cpp's own
  `whisper-quantize` (v1.9.4). Source and output SHA256s and sizes are recorded in
  `results/model-metadata.json`.

## Usage

```bash
python3 benchmarks/stt/run_benchmark.py --setup        # build images, download + quantize models
python3 benchmarks/stt/run_benchmark.py --skip-setup   # run the matrix
```

Outputs:

- `results/matrix.json` — full machine-readable results (includes raw transcripts)
- `results/matrix.md` — matrix table, medians, audio conditions, extrapolations
- `results/runs/*.json` — raw per-run payloads
- `results/model-metadata.json` — model hashes/sizes (safe to commit)

`matrix.json`, `matrix.md` and `runs/` contain speech content and are **git-ignored
on purpose**; only scripts, the reference text and `model-metadata.json` are committed.

Measuring the audio conditions (Phase 9) uses the application container's `ffmpeg`
read-only via `docker compose exec backend`, so the app container must be running.
The audio file itself is never modified or preprocessed.

## Scoring

`normalize.py` implements Turkish-aware normalization (Unicode NFC, lowercase,
punctuation removal, whitespace collapse, Turkish letters preserved) plus a
standard-library dynamic-programming WER with substitution/deletion/insertion
counts. Combined WER across samples uses total errors / total reference words,
never a simple average of per-sample WERs. No NLP dependency is used.
