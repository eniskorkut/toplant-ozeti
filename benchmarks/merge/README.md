# STT word timestamps + diarization merge (benchmark-only)

Validates the last technical risk before integration: **can word-level STT timestamps be
combined with sherpa-onnx diarization into a reliable speaker-attributed transcript?**

Nothing here touches the FastAPI application, the production images or the installed
models. All raw outputs (transcripts, segments, merged turns) are git-ignored.

## Pipeline

```text
recording
  ├── STT (whisper.cpp turbo Q8_0  |  faster-whisper turbo INT8, word_timestamps=True)
  ├── diarization (pyannote 3.0 + TitaNet Small, auto threshold 0.80; known count when available)
  └── merge (this directory) -> anonymous speaker turns + structural diagnostics
```

## whisper.cpp timestamp findings (v1.9.4, our build)

- Machine-readable timestamps come from `-ojf` (JSON full): every segment and every
  **word-piece token** carries `offsets` in milliseconds.
- `-nt` (no-timestamps) is **not** usable for the merge, and it measurably changed the
  decoded text on the Turkish reference recording (`Bugün ki` instead of `Bugünkü`,
  WER 0.1548 vs 0.1429). The merge harness therefore runs without `-nt`.
- Tokens are word-pieces; they are aggregated into words at space boundaries.
- `t_dtw = -1`: DTW token timestamps are **not** computed (they would require a separate
  DTW model that is not part of the benchmark assets), so token timestamps come from
  whisper.cpp's standard heuristic.
- Zero-duration tokens occur regularly (e.g. punctuation-only tokens); they are counted,
  and the merge resolves them via the midpoint rule.

## faster-whisper timestamps

- `word_timestamps=True` produces native word intervals (start/end per word).
- Enabling word timestamps costs speed: median RTF rises from ~0.32 (plain transcription)
  to ~0.40 on the benchmark recordings.

## Merge rules (implemented in `merge.py`, unit-tested)

1. assign the speaker with the largest temporal overlap;
2. for collapsed/zero-length intervals use the word midpoint's active speaker;
3. otherwise search only within a small boundary tolerance (default **250 ms**);
4. otherwise leave the word **unresolved** (`speaker = null`) and count it — never assign
   across a large gap, never invent a speaker.
5. consecutive words of one speaker form turns; a short turn is kept as its own turn
   (the diarization result is authoritative, nothing is absorbed into a neighbour).

Structural invariants are enforced and the run fails if any is violated: monotonic
timestamps, `start <= end`, non-negative times, turns sorted and non-empty, nothing
beyond the audio duration, unresolved words counted.

## Usage

```bash
python3 benchmarks/merge/tests/test_merge.py                      # 13 stdlib unit tests
python3 benchmarks/merge/run_merge_benchmark.py                   # all recordings
python3 benchmarks/merge/run_merge_benchmark.py --turkish-only
python3 benchmarks/merge/run_merge_benchmark.py --voxconverse-only
python3 benchmarks/merge/run_merge_benchmark.py --force            # ignore STT cache
```

Outputs (git-ignored): `results/merge.json`, `results/merge.md`, `results/raw/*.json`.
