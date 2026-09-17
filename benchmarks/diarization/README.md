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

## VoxConverse 0.3 ground-truth benchmark (current round)

Scores both embeddings against real multi-speaker audio with reference RTTM annotations.

- Dataset: **VoxConverse 0.3** — dev set audio from the official Oxford download
  (`voxconverse_dev_wav.zip`), annotations from `joonson/voxconverse` master. License
  CC BY 4.0, research purposes. The archive is verified after download (exact size,
  full `zipfile.testzip()` integrity pass, SHA256 recorded in `provenance.json`).
- Subset: 12 recordings selected deterministically from RTTMs before any model ran
  (4 with 2 speakers, 4 with 3, 4 with ≥4; shortest valid files first, ≥30 s each).
  Ids live in `voxconverse_subset.txt`; calibration/validation split is the first two
  vs. the last two files of each group.
- Scoring: isolated `nryant/dscore` checkout pinned to commit
  `e02f949ac6592279300a2c33d03daf9e0c12fd27` (`dscore.Dockerfile`), collar **0.25 s**,
  overlaps **included** for the primary DER and a diagnostic DER with overlaps ignored.
  Global DER/JER come from dscore's multi-file scoring (time-weighted), never an average
  of per-file percentages. Per-file missed speech / false alarm / speaker error are not
  exposed by this dscore version and are reported as unavailable.
- System RTTM output uses anonymous `speaker_<n>` labels only; dscore performs the
  speaker mapping. Reference identities are never copied into system output.

Stages (resumable, state in `results/voxconverse-state.json`, git-ignored):

```bash
python3 benchmarks/diarization/download_voxconverse.py      # parallel download + integrity gate
python3 benchmarks/diarization/select_voxconverse_subset.py # writes voxconverse_subset.txt
python3 benchmarks/diarization/run_voxconverse.py --mode setup
python3 benchmarks/diarization/run_voxconverse.py --mode known      # num_clusters = reference count
python3 benchmarks/diarization/run_voxconverse.py --mode calibrate   # sweep 0.30 … 0.65
python3 benchmarks/diarization/run_voxconverse.py --mode validate    # held-out split
python3 benchmarks/diarization/run_voxconverse.py --mode repeat      # 3 repetitions
python3 benchmarks/diarization/run_voxconverse.py --mode report      # voxconverse.{json,md}
```

Threshold selection follows the documented order (lowest calibration DER, then JER,
then count MAE, then closest to 0.50); values within 1e-6 are treated as ties so
floating-point noise cannot override the tie breakers. Validation data is never used
for tuning, and note the selected thresholds sit at the upper end of the allowed sweep.

## Earlier round: two-speaker playback-through-microphone test

That test is a smoke/consistency check only and must not be used for model selection.

### Playback test usage


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
