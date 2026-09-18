# ElevenLabs Scribe v2 benchmark (isolated, opt-in)

Compares ElevenLabs `scribe_v2` (word timestamps + diarization) against the local
whisper.cpp/TitaNet pipeline on **existing recordings only**. Nothing here touches the
application, the runtime database or the local benchmark artifacts.

## Privacy and security rules

- The key is read from the `ELEVENLABS_API_KEY` environment variable and is passed to
  `curl` through a config on stdin — it never appears in argv, in a log line, in a file
  or in the committed aggregate.
- **Normal ElevenLabs retention may apply:** the current account is a normal/free
  development account, so Zero Retention Mode is not requested and `enable_logging` is
  left untouched.
- Free-plan usage is limited: the harness enforces a hard budget of **6 requests**
  (one requested benchmark call = one network call, no automatic retries) and stops
  immediately if quota is reported insufficient.
- Raw responses and private transcripts stay in the git-ignored `results/` directory.

## Where the key goes

Two options, both kept out of git:

1. **Environment variable** (preferred): `export ELEVENLABS_API_KEY=...` in the shell
   that runs the harness. The value is never logged or written anywhere.
2. **Git-ignored harness-local file**: `benchmarks/elevenlabs-stt/.env.local`
   (exact path `/Users/<you>/Desktop/toplantı/benchmarks/elevenlabs-stt/.env.local`),
   containing a single line `ELEVENLABS_API_KEY=...`. The file is ignored by the root
   `.env.*` rule; the harness reads it only when the environment variable is absent.

> `frontend/.env.local` is **not** a valid location for this key: the frontend runs in
> the browser, and the benchmark harness runs on the host.

## Check the key first

```bash
python3 benchmarks/elevenlabs-stt/run_benchmark.py --check-key
# prints only: elevenlabs_key_configured = true|false
```

If it prints `false`, **no request is made** and the benchmark cannot run.

## Calls (exactly six, in this order)

| Call | Audio | Fields |
|---|---|---|
| `real-known4` | real four-speaker meeting | `tur`, diarize, `num_speakers=4` |
| `real-auto` | same audio | `tur`, diarize (automatic count) |
| `turkish-far` | controlled reference recording | `tur`, no diarization |
| `turkish-near` | controlled reference recording | `tur`, no diarization |
| `voxconverse-whmpa` | VoxConverse validation file | `eng`, diarize, `num_speakers=2` |
| `voxconverse-wjhgf` | VoxConverse validation file | `eng`, diarize, `num_speakers=5` |

Common fields: `model_id=scribe_v2`, `timestamps_granularity=word`,
`tag_audio_events=false`, `no_verbatim=false`. No entity detection, keyterm prompting,
speaker library, speaker roles, multi-channel or webhooks.

```bash
python3 benchmarks/elevenlabs-stt/run_benchmark.py --call real-known4
```

Each call writes a private transcript to `results/private/` and a metrics-only summary to
stdout; `results/request-guard.json` records the request count and total audio seconds
uploaded.

## Local comparison data (no inference)

`local_spot_check.py` (backend image) prints the per-file local numbers for the same
VoxConverse files from the stored sweep segments, so the fair comparison does not spend
CPU time re-running diarization:

```bash
cd benchmarks/elevenlabs-stt
docker compose -f ../../compose.yaml run --rm -e PYTHONPATH=/app \
    -v "$PWD:/bench:ro" -v "$PWD/results:/out" -v "$PWD/../overlap-merge:/overlap:ro" \
    -v "$PWD/../overlap-merge/results/raw:/stt-raw:ro" \
    -v "$PWD/../diarization-postprocess/results/segments/validation:/sweep-segments:ro" \
    -v "$PWD/../diarization/datasets/voxconverse:/voxconverse:ro" \
    backend uv run --locked python /bench/local_spot_check.py
```

## Results (measured, six calls, 310.5 s of audio uploaded)

| Metric | Local whisper.cpp | ElevenLabs scribe_v2 |
|---|---:|---:|
| Turkish WER (far / near / combined) | 0.1429 / 0.2024 / **0.1726** | 0.0476 / 0.0119 / **0.0298** |

Real four-speaker meeting (no temporal ground truth): local production assigns 85/106
words, tuned 0/0 assigns 91/106, ElevenLabs tags 99/99 (known4) and 110/110 (auto) — but
ElevenLabs clusters only **2 speakers** (known4, despite `num_speakers=4`) / 3 (auto),
where local diarization returns the expected 4. Latency: ElevenLabs 4.2–4.9 s vs local
STT 14.4 s + diarization.

Limited VoxConverse spot check (best-permutation agreement / wrong attribution):
`whmpa` local 0.9281 / 7.19% vs ElevenLabs **0.9724 / 2.76%**; `wjhgf` local 0.7626 /
23.74% vs ElevenLabs **0.8111 / 18.89%**.

Retention: normal ElevenLabs retention may apply (free/dev account, no zero-retention
mode requested).

## Tests

```bash
python3 benchmarks/elevenlabs-stt/tests/test_elevenlabs.py
```

Twenty standard-library tests cover response parsing (`type == "word"` only), word
filtering, speaker grouping, missing `speaker_id`, timestamp validation, rapid flips,
error redaction, the request budget and the guarantee that the metrics summary never
contains transcript text. **No test performs a network call.**
