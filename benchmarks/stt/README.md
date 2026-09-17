# STT benchmark environment (temporary, isolated)

Purpose: measure two serious CPU speech-to-text candidates on the same real
Turkish recording before choosing a production implementation.

**Nothing here is part of the application runtime.**

- No STT dependency is added to `backend/pyproject.toml`.
- No STT build tool is added to the production backend image.
- Both candidates run in their own Docker images, CPU only, Linux, Python 3.12 host tooling.
- No MLX / Metal / CoreML / CUDA / ROCm / Vulkan.

## Candidates

| | Candidate A | Candidate B |
|---|---|---|
| Engine | faster-whisper (`SYSTRAN/faster-whisper`) 1.2.1 | whisper.cpp (`ggml-org/whisper.cpp`) v1.9.4 |
| Runtime | CTranslate2 CPU, `compute_type="int8"` | `whisper-cli`, ggml quantized |
| Model | `small` (multilingual) | `small-q5_1` (official distribution) |
| Language | `tr` | `tr` |
| Decoding | `beam_size=5`, no batching | `-bs 5` |
| VAD | off | off |
| Threads | 8 (identical for both) | 8 (identical for both) |

## Fairness rules implemented

- Both candidates read the **same** `processing.wav` (mounted read-only at `/audio`).
- Same language, same thread count, same beam size as closely as the
  implementations permit, VAD disabled for both.
- No image build time, no model download time, no warmup run is included in the
  measured statistics.
- One warmup run + three measured runs per candidate.
- Peak RSS is captured with GNU `/usr/bin/time -v` inside the container.
- Model caches live in `benchmarks/stt/.cache/` (git-ignored, not committed).

## Usage

```bash
python3 benchmarks/stt/run_benchmark.py --setup            # build images + download models
python3 benchmarks/stt/run_benchmark.py --threads 8        # warmup + 3 measured runs per candidate
python3 benchmarks/stt/run_benchmark.py --audio data/meetings/<id>/processing.wav
```

`--audio` defaults to the newest `data/meetings/*/processing.wav`.

Outputs:

- `results/latest.json` — full machine-readable results (includes raw transcripts)
- `results/latest.md` — comparison table, extrapolations, raw transcripts
- `results/runs/*.json` — raw per-run payloads

`results/latest.json`, `results/latest.md` and `results/runs/` contain speech
content and are **git-ignored on purpose**; only the scripts and the reference
text are committed.

## Scoring

`normalize.py` implements Turkish-aware normalization (Unicode NFC, lowercase,
punctuation removal, whitespace collapse, Turkish letters preserved) plus a
standard-library dynamic-programming WER with substitution/deletion/insertion
counts. No NLP dependency is used.
