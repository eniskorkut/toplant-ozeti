# Diarization post-processing sweep (min_duration_on / min_duration_off)

Fixed: sherpa-onnx 1.10.46, pyannote segmentation 3.0, TitaNet Small, automatic speaker
count, `threshold=0.80`, 8 threads, production merge rules and the current heuristic
whisper.cpp timestamps. Only the two post-processing parameters change.

Grid (exactly 16 configurations):

| min_duration_on | 0.0 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|
| min_duration_off | 0.00 | 0.10 | 0.25 | 0.50 |

Production defaults (`0.3 / 0.5`) are the baseline; nothing in production is modified.

## API semantics (verified against the installed 1.10.46 build)

`OfflineSpeakerDiarizationConfig` accepts float `min_duration_on` / `min_duration_off`
with built-in defaults `0.3` / `0.5` (the on value round-trips as float32). No
`window_shift_ratio` is tested — that option does not exist in this pinned version.

## Pipeline

1. `run_sweep.py` (backend image) — diarizes every configuration × file, stores segments
   and system RTTMs, and computes coverage, overlap, segment-duration buckets,
   production-merge diagnostics (assigned/unresolved/causes, turns, rapid flips) and
   best-permutation word→speaker agreement against the reference RTTMs.
2. `score_sweep.py` (host) — isolated dscore image, collar 0.25 s, overlaps included and
   a diagnostic overlaps-ignored pass, aggregated per configuration.
3. `finalize.py` (host) — applies the safety-aware selection and writes the reports.

## Selection rules (`selection.py`)

False attribution is worse than an unresolved word. A candidate is only eligible when its
wrong-attribution rate does not exceed the production baseline by more than **0.005
absolute**; among eligible candidates the ranking is: lowest combined assignment error
`(wrong + unresolved) / total words` → lowest DER → lowest unresolved rate → lowest
speaker-count MAE → closest parameters to `0.3 / 0.5`. Validation data is never used for
selection.

## Running

```bash
cd benchmarks/diarization-postprocess
docker compose -f ../../compose.yaml run --rm -e PYTHONPATH=/app \
    -v "$PWD:/bench" -v "$PWD/../overlap-merge:/overlap:ro" \
    -v "$PWD/../overlap-merge/results/raw:/stt-raw:ro" \
    -v "$PWD/../diarization/datasets/voxconverse:/voxconverse:ro" \
    -v "$PWD/../diarization/models:/models/diarization:ro" \
    -v "$PWD/../../data:/data:ro" \
    backend uv run --locked python /bench/run_sweep.py --stage calibration
python3 score_sweep.py --stage calibration
python3 finalize.py --stage calibration
# then --stage validation with --configs "<baseline>,<selected>" and --stage real
python3 tests/test_postprocess.py
```

Committed: harness, tests, README and the text-free `aggregate-*.json` reports.
Git-ignored: `results/` (segments, system RTTMs, private transcripts, detailed reports).
