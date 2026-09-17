# Speaker diarization benchmark (sherpa-onnx 1.10.46)

Purpose: determine whether the **already installed** sherpa-onnx stack can reliably
detect speaker count, assign stable anonymous speaker ids, survive short turns, and
run efficiently on CPU — before any production integration.

**Test type for the current round: two-speaker playback-through-microphone test.**

> Not equivalent to a real physical multi-speaker meeting benchmark: the audio is a
> two-speaker conversation played back through a device and captured by the browser
> microphone pipeline. A proper physical multi-speaker benchmark should follow.

## Isolation

- Runs **inside the existing backend image** (`meeting-intelligence-backend:dev`),
  which already contains `sherpa-onnx==1.10.46`.
- **No new Python dependency** is added to `backend/pyproject.toml`.
- No application code, FastAPI route or production configuration is touched.
- Models are cached under `benchmarks/diarization/models/` (git-ignored).

## Models (official k2-fsa/sherpa-onnx releases only)

| Role | Asset | Verification |
|---|---|---|
| Segmentation | `sherpa-onnx-pyannote-segmentation-3-0` | SHA256 of archive + extracted `model.onnx` recorded (no official checksum published) |
| D1 embedding | `3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx` | SHA256 matches the official `checksum.txt` |
| D2 embedding | `nemo_en_titanet_small.onnx` | SHA256 matches the official `checksum.txt` |

**INT8 embedding variants are not published** by the official project (verified by
listing the release assets), so the official FP32 models are used. Nothing is patched.

## Installed-API notes (1.10.46, verified by introspection)

- `OfflineSpeakerDiarization`, `OfflineSpeakerDiarizationConfig`,
  `OfflineSpeakerSegmentationModelConfig`, `OfflineSpeakerSegmentationPyannoteModelConfig`,
  `SpeakerEmbeddingExtractorConfig`, `FastClusteringConfig` all exist.
- `OfflineSpeakerDiarizationConfig(segmentation, embedding, clustering, min_duration_on=0.3, min_duration_off=0.5)`.
- **`window_shift_ratio` does not exist in 1.10.46** — the segmentation window shift is
  internal to the pyannote model. Nothing is silently substituted; this is reported.
- `min_duration_on=0.3` / `min_duration_off=0.5` are the 1.10.46 defaults and are set
  explicitly to the requested values.
- `sherpa_onnx.read_wave` does not exist in 1.10.46; the engine reads PCM16 mono WAV
  with the standard-library `wave` module (no resampling, no preprocessing).

## Modes

- **known count** — `num_clusters = N` (control run; for the current round `N = 2`),
  1 warmup + 3 measured runs per embedding model.
- **automatic count** — `num_clusters = -1` with a threshold sweep
  `0.50 / 0.65 / 0.80 / 0.90`, one run per threshold.

## Metrics (and what is deliberately not reported)

- Structural: detected speaker count, segments, turns after merging consecutive
  same-speaker segments, speech duration per cluster, short turns (< 1 s).
- Sequence: with an operator-provided approximate turn order, anonymous clusters are
  mapped to the reference labels with the **best permutation after inference** (never
  renamed by hand), then scored with a global sequence alignment: matches, missed
  (deleted) reference turns, merged errors, split errors, turn consistency.
- **No DER is reported.** Exact temporal ground truth cannot be derived reliably from
  the recording, so no DER is invented and Whisper/STT is not used for scoring.

## Reference turn order (optional)

Write the approximate turn order to `benchmarks/diarization/reference_turns.txt`, one
label per turn, e.g.:

```text
A B A B A B
```

If the file is absent, sequence metrics are reported as `n/a` and only structural
metrics are produced. The file contains no speech text and is git-ignored because it
describes a specific recording session.

## Usage

```bash
python3 benchmarks/diarization/run_benchmark.py --setup            # download + verify official models
python3 benchmarks/diarization/run_benchmark.py --skip-setup \
    --audio data/meetings/<id>/processing.wav \
    --expected-speakers 2
```

Outputs (git-ignored, may contain meeting information):

- `results/latest.json` — full results including segment timings
- `results/latest.md` — result matrix, threshold sweep, performance, metric labels
- `results/runs/*.json` — raw per-run payloads

Committed metadata: `results/model-metadata.json` (model names, sizes, SHA256 only).

## Resource notes

Diarization runs in short-lived containers; peak RSS is measured with
`resource.getrusage(RUSAGE_SELF).ru_maxrss` inside the engine process (no extra tooling
needed in the production image).
