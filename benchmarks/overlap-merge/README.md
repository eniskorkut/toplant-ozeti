# Overlap-aware speaker merge benchmark (DTW timestamps + conservative resolver)

Benchmark-only investigation triggered by the real four-speaker meeting
(`b1095740120b4b1e96db337da961ea63`, 42.84 s, 4 real speakers) where 21 of 106 words
were left `Bilinmeyen`.

## Configurations

| Config | STT timestamps | Merge |
|---|---|---|
| M0 | whisper.cpp heuristic token offsets (production: flash attention ON) | production baseline |
| M0n | heuristic offsets, flash attention OFF (control) | production baseline |
| M1 | **DTW** `-dtw large.v3.turbo` | production baseline |
| M2 | DTW | baseline first, then the conservative context resolver |

Production decoding flags are unchanged (large-v3-turbo Q8_0, `-l tr`, `-t 8`, `-bs 5`,
`-ojf`, no `-nt`); DTW only adds `-nfa` because our v1.9.4 build disables DTW when
flash attention is on.

## Findings that shaped the harness

- `-dtw large.v3.turbo` **is supported** in our v1.9.4 build (`examples/cli/cli.cpp`
  maps the preset to the compiled-in `WHISPER_AHEADS_LARGE_V3_TURBO`; no extra model
  asset is needed).
- The default CLI has `-fa true`, and `whisper.cpp` then logs
  `dtw_token_timestamps is not supported with flash_attn - disabling`. DTW therefore
  requires `-nfa`.
- Disabling flash attention changes the decoded text (106 → 96 words on the real
  meeting); DTW itself does not (heuristic-`-nfa` and DTW texts are identical). M0n
  exists to separate those effects.

## Conservative resolver (`context_resolver.py`)

Only words the baseline left unresolved are considered.

- **Case A (overlap ambiguity):** sum each speaker's active time in a local window
  around the word midpoint (radii 0.25 / 0.50 / 0.75 / 1.00 s) and resolve only on a
  unique highest support that beats the runner-up by a margin (0.10 / 0.20 / 0.30 s).
  Exact ties are never broken by speaker ordering.
- **Case B (coverage gap):** bridge only when the previous and next resolved words share
  a speaker, the word lies between them, and diarization activity is within 0.50 s.
  Short words never automatically inherit a neighbour.

## Calibration and validation

The radius/margin grid is calibrated on the VoxConverse **calibration split** only
(6 files, reference RTTMs, TitaNet @0.80 diarization reused from the diarization
benchmark) with the selection rule: lowest wrong-attribution rate → highest agreement →
lowest unresolved rate → smallest radius → largest margin. The selected configuration is
then validated once on the held-out split. The private four-speaker meeting is never used
for tuning.

## Running

```bash
python3 benchmarks/overlap-merge/run_stt.py          # cached whisper outputs (heuristic, heuristic-nfa, dtw)
cd benchmarks/overlap-merge && docker compose -f ../../compose.yaml run --rm \
    -e PYTHONPATH=/app -v "$PWD:/bench" \
    -v "$PWD/../diarization/datasets/voxconverse:/voxconverse:ro" \
    -v "$PWD/../diarization/results:/diar-results:ro" \
    -v "$PWD/../diarization/models:/models/diarization:ro" \
    backend uv run --locked python /bench/analyze.py
python3 benchmarks/overlap-merge/tests/test_overlap_merge.py
```

Committed: `aggregate.json` (metrics only, no transcript text), harness, tests, README.
Git-ignored: `results/` (raw STT outputs, private transcripts, attribution diff).
