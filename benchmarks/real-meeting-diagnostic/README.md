# Real four-speaker meeting diagnostic

Controlled A/B/C/D diagnostic against **one existing real recording** — the same
`processing.wav` is reused read-only for every run so the input is controlled. The
recording, its meeting row, transcript, analysis, MP3 and WAV are never modified.

| Config | STT language | Diarization |
|---|---|---|
| A | `auto` | automatic (`num_clusters=-1`, threshold 0.80) |
| B | `tr` | automatic (threshold 0.80) |
| C | `auto` | known count (`num_clusters=4`) |
| D | `tr` | known count (`num_clusters=4`) |

Everything else stays at production defaults (whisper.cpp v1.9.4 large-v3-turbo Q8_0,
8 threads, `-ojf`, beam 5; sherpa-onnx 1.10.46 + pyannote 3.0 + TitaNet Small,
`min_duration_on=0.3`, `min_duration_off=0.5`; production merge rules).

## Running

```bash
docker compose run --rm -e PYTHONPATH=/app \
    -v "$PWD/benchmarks/real-meeting-diagnostic:/bench" \
    backend uv run --locked python /bench/run_diagnostic.py \
        --meeting-id <meeting-id> \
        --audio /data/meetings/<meeting-id>/processing.wav \
        --real-speakers 4
```

Outputs:

- `aggregate.json` (committed) — aggregate metrics only, **no transcript text**;
- `output/` (git-ignored) — `A-auto-auto.txt`, `B-tr-auto.txt`, `C-auto-known4.txt`,
  `D-tr-known4.txt` (`[start] Speaker: text`), `diff_A_vs_B.txt`, `diff_C_vs_D.txt`,
  the full aggregate including short observational snippets, and `aggregate.md`.

Metrics helpers live in `diagnostic_metrics.py` and are unit-tested without any ML
runtime:

```bash
python3 benchmarks/real-meeting-diagnostic/tests/test_metrics.py
```
