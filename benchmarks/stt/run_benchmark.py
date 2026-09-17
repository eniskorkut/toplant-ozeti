#!/usr/bin/env python3
"""Reproducible CPU STT benchmark matrix.

Configurations (all CPU-only, VAD off, language tr, beam 5, 8 threads):
    A1  faster-whisper small int8, sequential (batch size 1)
    A2  faster-whisper small int8, official BatchedInferencePipeline (batch size 8)
    B1  whisper.cpp small q5_1
    B2  whisper.cpp small q8_0

Samples:
    sample_far   existing quiet/distant recording
    sample_near  controlled close recording of the same reference text

Usage:
    python3 benchmarks/stt/run_benchmark.py --setup
    python3 benchmarks/stt/run_benchmark.py --skip-setup
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

STT_DIR = Path(__file__).resolve().parent
REPO_ROOT = STT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "meetings"
CACHE_DIR = STT_DIR / ".cache"
RESULTS_DIR = STT_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"

FASTER_WHISPER_DIR = STT_DIR / "faster-whisper"
WHISPER_CPP_DIR = STT_DIR / "whisper-cpp"
FASTER_WHISPER_IMAGE = "mi-bench-faster-whisper:local"
WHISPER_CPP_IMAGE = "mi-bench-whisper-cpp:local"

FAR_RECORDING_ID = "fcdf1737901d41ddacd14429d39a4183"

CONTAINER_AUDIO_ROOT = "/audio"
APP_CONTAINER_DATA_ROOT = "/data/meetings"
WHISPER_CPP_MODEL_DIR = "/models"

MAX_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
WALL_CLOCK_RE = re.compile(r"Elapsed \(wall clock\) time .*?: (\d+):(\d+(?:\.\d+)?)")
MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[\d.]+) dB")
MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?[\d.]+) dB")


def log(message: str) -> None:
    print(message, flush=True)


def docker(
    args: list[str], *, capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=capture, text=True, check=check, stdin=subprocess.DEVNULL
    )


def run_in_container(
    image: str,
    command: list[str],
    *,
    volumes: list[str],
    environment: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    args = ["run", "--rm"]
    for volume in volumes:
        args += ["-v", volume]
    for key, value in environment.items():
        args += ["-e", f"{key}={value}"]
    args += [image, *command]
    return docker(args, check=check)


def host_arch() -> str:
    machine = platform.machine()
    return {"arm64": "aarch64"}.get(machine, machine)


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time_output(stderr: str) -> tuple[int | None, float | None]:
    rss_match = MAX_RSS_RE.search(stderr)
    wall_match = WALL_CLOCK_RE.search(stderr)
    peak_rss_kb = int(rss_match.group(1)) if rss_match else None
    wall_seconds = (
        float(wall_match.group(1)) * 60 + float(wall_match.group(2)) if wall_match else None
    )
    return peak_rss_kb, wall_seconds


def parse_engine_json(stdout: str) -> dict:
    for line in reversed(stdout.strip().splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return json.loads(candidate)
    raise RuntimeError(f"no JSON payload found in container output:\n{stdout[-2000:]}")


def container_audio_path(audio: Path) -> str:
    relative = audio.resolve().relative_to(DATA_DIR.resolve())
    return f"{CONTAINER_AUDIO_ROOT}/{relative.as_posix()}"


def resolve_samples(far_audio: Path | None, near_audio: Path | None) -> dict[str, Path]:
    far = far_audio or DATA_DIR / FAR_RECORDING_ID / "processing.wav"
    if not far.exists():
        raise SystemExit(f"sample_far not found: {far}")

    if near_audio is not None:
        near = near_audio
    else:
        candidates = sorted(
            (
                path
                for path in DATA_DIR.glob("*/processing.wav")
                if path.parent.name != FAR_RECORDING_ID
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit("sample_near not found: record the reference text again first")
        near = candidates[0]

    if not near.exists():
        raise SystemExit(f"sample_near not found: {near}")
    return {"sample_far": far.resolve(), "sample_near": near.resolve()}


def audio_conditions(audio: Path) -> dict:
    """Measure level without modifying the audio, using the app container's ffmpeg."""
    app_relative = audio.resolve().relative_to(DATA_DIR.resolve()).as_posix()
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "backend",
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-i",
            f"{APP_CONTAINER_DATA_ROOT}/{app_relative}",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    stderr = completed.stderr
    mean_match = MEAN_VOLUME_RE.search(stderr)
    max_match = MAX_VOLUME_RE.search(stderr)
    maximum = float(max_match.group(1)) if max_match else None

    return {
        "path": str(audio.relative_to(REPO_ROOT)),
        "duration_seconds": round(wav_duration_seconds(audio), 3),
        "size_bytes": audio.stat().st_size,
        "sha256": sha256_of(audio),
        "mean_volume_db": float(mean_match.group(1)) if mean_match else None,
        "max_volume_db": maximum,
        "clipping": bool(maximum is not None and maximum >= -0.1),
    }


def build_images() -> None:
    log("building faster-whisper benchmark image (faster-whisper 1.2.1)")
    docker(["build", "-t", FASTER_WHISPER_IMAGE, str(FASTER_WHISPER_DIR)])
    log("building whisper.cpp benchmark image (v1.9.4, CPU only)")
    docker(["build", "-t", WHISPER_CPP_IMAGE, str(WHISPER_CPP_DIR)])


def download_models() -> None:
    (CACHE_DIR / "huggingface").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "whisper-cpp").mkdir(parents=True, exist_ok=True)

    log("downloading faster-whisper model: small (benchmark cache only)")
    run_in_container(
        FASTER_WHISPER_IMAGE,
        [
            "python",
            "-c",
            (
                "from faster_whisper import WhisperModel;"
                "WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=8);"
                "print('cached')"
            ),
        ],
        volumes=[f"{CACHE_DIR / 'huggingface'}:/cache/huggingface"],
        environment={"HF_HOME": "/cache/huggingface"},
    )

    for model in ("small-q5_1", "small"):
        log(f"downloading official whisper.cpp model: {model}")
        run_in_container(
            WHISPER_CPP_IMAGE,
            ["/opt/whisper.cpp/models/download-ggml-model.sh", model, WHISPER_CPP_MODEL_DIR],
            volumes=[f"{CACHE_DIR / 'whisper-cpp'}:{WHISPER_CPP_MODEL_DIR}"],
            environment={},
        )


def prepare_q8_0_model() -> dict:
    models = CACHE_DIR / "whisper-cpp"
    source = models / "ggml-small.bin"
    target = models / "ggml-small-q8_0.bin"
    if not source.exists():
        raise SystemExit(f"official source model missing: {source}")

    if not target.exists():
        log("quantizing ggml-small.bin -> ggml-small-q8_0.bin (whisper.cpp whisper-quantize, q8_0)")
        run_in_container(
            WHISPER_CPP_IMAGE,
            [
                "whisper-quantize",
                f"{WHISPER_CPP_MODEL_DIR}/ggml-small.bin",
                f"{WHISPER_CPP_MODEL_DIR}/ggml-small-q8_0.bin",
                "q8_0",
            ],
            volumes=[f"{models}:{WHISPER_CPP_MODEL_DIR}"],
            environment={},
        )

    metadata = {
        "source": "ggml-small.bin (official whisper.cpp distribution)",
        "source_sha256": sha256_of(source),
        "source_size_bytes": source.stat().st_size,
        "quantized": "ggml-small-q8_0.bin",
        "quantized_sha256": sha256_of(target),
        "quantized_size_bytes": target.stat().st_size,
        "quantization": "q8_0",
        "tool": "whisper.cpp quantize (v1.9.4)",
    }
    (CACHE_DIR / "model-metadata-small.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "model-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    log(
        f"q8_0 model ready ({metadata['quantized_size_bytes']} bytes, sha256 "
        f"{metadata['quantized_sha256'][:16]}…)"
    )
    return metadata


def faster_whisper_repo_id(model: str) -> str:
    """The official CT2 repo id that faster-whisper itself maps a model name to."""
    output = docker(
        [
            "run",
            "--rm",
            "--entrypoint",
            "python",
            FASTER_WHISPER_IMAGE,
            "-c",
            f"from faster_whisper.utils import _MODELS;print(_MODELS['{model}'])",
        ]
    ).stdout.strip()
    return output


def curl_json(url: str) -> object:
    completed = subprocess.run(
        ["curl", "-fsSL", url],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return json.loads(completed.stdout)


def hf_file_info(repo: str, filename: str) -> dict:
    """Expected size and sha256 (LFS oid) for a file in an official HF repo."""
    entries = curl_json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=1")
    for entry in entries:
        if entry.get("path") == filename:
            lfs = entry.get("lfs") or {}
            return {
                "size": entry.get("size") or lfs.get("size"),
                "sha256": lfs.get("oid"),
            }
    raise SystemExit(f"{filename} not found in official repo {repo}")


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def vm_memory_kb(image: str) -> dict[str, int]:
    output = docker(["run", "--rm", "--entrypoint", "cat", image, "/proc/meminfo"]).stdout
    values: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0])
    return values


def download_turbo_models() -> dict:
    """Download both official turbo models, verify them, and record resource metadata."""
    (CACHE_DIR / "huggingface").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "whisper-cpp").mkdir(parents=True, exist_ok=True)

    log("downloading faster-whisper model: large-v3-turbo (download only, no model load)")
    fw_repo = faster_whisper_repo_id("large-v3-turbo")
    run_in_container(
        FASTER_WHISPER_IMAGE,
        [
            "python",
            "-c",
            (
                "from huggingface_hub import snapshot_download;"
                f"snapshot_download(repo_id='{fw_repo}');print('cached')"
            ),
        ],
        volumes=[f"{CACHE_DIR / 'huggingface'}:/cache/huggingface"],
        environment={"HF_HOME": "/cache/huggingface"},
    )

    log("downloading official whisper.cpp model: large-v3-turbo-q8_0")
    run_in_container(
        WHISPER_CPP_IMAGE,
        [
            "/opt/whisper.cpp/models/download-ggml-model.sh",
            "large-v3-turbo-q8_0",
            WHISPER_CPP_MODEL_DIR,
        ],
        volumes=[f"{CACHE_DIR / 'whisper-cpp'}:{WHISPER_CPP_MODEL_DIR}"],
        environment={},
    )

    # Official source of truth for the whisper.cpp model, verified before benchmarking.
    cpp_model = CACHE_DIR / "whisper-cpp" / "ggml-large-v3-turbo-q8_0.bin"
    if not cpp_model.exists():
        raise SystemExit(f"whisper.cpp turbo model missing after download: {cpp_model}")
    cpp_expected = hf_file_info("ggerganov/whisper.cpp", "ggml-large-v3-turbo-q8_0.bin")
    cpp_actual_sha = sha256_of(cpp_model)
    cpp_actual_size = cpp_model.stat().st_size
    cpp_verified = (
        cpp_expected["sha256"] == cpp_actual_sha and cpp_expected["size"] == cpp_actual_size
    )
    if not cpp_verified:
        raise SystemExit(
            "official model verification failed for ggml-large-v3-turbo-q8_0.bin: "
            f"expected size={cpp_expected['size']} sha256={cpp_expected['sha256']}, "
            f"actual size={cpp_actual_size} sha256={cpp_actual_sha}"
        )

    fw_repo_dir = CACHE_DIR / "huggingface" / "hub" / f"models--{fw_repo.replace('/', '--')}"
    fw_models = sorted(fw_repo_dir.glob("snapshots/*/model.bin"))
    fw_metadata: dict = {"repo": fw_repo, "resolved_by": "faster_whisper.utils._MODELS"}
    if fw_models:
        fw_model = fw_models[0]
        fw_expected = hf_file_info(fw_repo, "model.bin")
        fw_actual_sha = sha256_of(fw_model)
        fw_metadata.update(
            {
                "model_bin_size_bytes": fw_model.stat().st_size,
                "model_bin_sha256": fw_actual_sha,
                "expected_size_bytes": fw_expected["size"],
                "expected_sha256": fw_expected["sha256"],
                "verified": fw_expected["sha256"] == fw_actual_sha
                and fw_expected["size"] == fw_model.stat().st_size,
            }
        )
    fw_metadata["cache_size_bytes"] = directory_size_bytes(fw_repo_dir)

    memory = vm_memory_kb(FASTER_WHISPER_IMAGE)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "faster_whisper": fw_metadata,
        "whisper_cpp": {
            "model": "ggml-large-v3-turbo-q8_0.bin",
            "source": "official ggerganov/whisper.cpp distribution",
            "size_bytes": cpp_actual_size,
            "sha256": cpp_actual_sha,
            "expected_size_bytes": cpp_expected["size"],
            "expected_sha256": cpp_expected["sha256"],
            "verified": cpp_verified,
            "locally_quantized": False,
        },
        "container_memory": {
            "mem_total_kb": memory.get("MemTotal"),
            "swap_total_kb": memory.get("SwapTotal"),
            "swap_free_kb": memory.get("SwapFree"),
        },
    }
    (RESULTS_DIR / TIERS["turbo"]["metadata"]).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    log(
        "turbo models verified: whisper.cpp q8_0 sha256 "
        f"{cpp_actual_sha[:16]}… ({cpp_actual_size} bytes), faster-whisper cache "
        f"{fw_metadata['cache_size_bytes']} bytes"
    )
    return metadata


SMALL_CONFIGS: list[dict] = [
    {"id": "fw-small-int8-b1", "label": "FW small INT8 batch1", "engine": "faster-whisper", "batch_size": 1},
    {"id": "fw-small-int8-b8", "label": "FW small INT8 batch8", "engine": "faster-whisper", "batch_size": 8},
    {"id": "wc-small-q5_1", "label": "whisper.cpp small Q5_1", "engine": "whisper.cpp", "model": "small-q5_1"},
    {"id": "wc-small-q8_0", "label": "whisper.cpp small Q8_0", "engine": "whisper.cpp", "model": "small-q8_0"},
]

# Turbo tier: sequential decoding only for the faster-whisper candidate (the
# previous round showed batching changes context handling and worsens WER).
TURBO_CONFIGS: list[dict] = [
    {
        "id": "fw-large-v3-turbo-int8",
        "label": "FW large-v3-turbo INT8",
        "engine": "faster-whisper",
        "model": "large-v3-turbo",
        "batch_size": 1,
    },
    {
        "id": "wc-large-v3-turbo-q8_0",
        "label": "whisper.cpp large-v3-turbo Q8_0",
        "engine": "whisper.cpp",
        "model": "large-v3-turbo-q8_0",
    },
]

TIERS: dict[str, dict] = {
    "small": {
        "configs": SMALL_CONFIGS,
        "json": "matrix.json",
        "markdown": "matrix.md",
        "runs": "runs",
        "metadata": "model-metadata.json",
    },
    "turbo": {
        "configs": TURBO_CONFIGS,
        "json": "turbo-matrix.json",
        "markdown": "turbo-matrix.md",
        "runs": "turbo-runs",
        "metadata": "model-metadata-turbo.json",
    },
}

# Historical small-tier baselines (used only when results/matrix.json is absent).
SMALL_BASELINES: list[dict] = [
    {"label": "FW small INT8 batch1", "wer": 0.2560, "rtf": 0.4413, "rss_mb": 760.2},
    {"label": "whisper.cpp small Q8_0", "wer": 0.3095, "rtf": 0.1215, "rss_mb": 601.4},
]

WER_BANDS = [
    (0.10, "excellent candidate"),
    (0.15, "strong candidate"),
    (0.20, "possibly usable, review errors"),
    (float("inf"), "not acceptable as final production quality"),
]

RTF_BANDS = [
    (0.15, "excellent CPU speed"),
    (0.30, "good for post-meeting processing"),
    (0.50, "acceptable but slower"),
    (float("inf"), "too slow for our preferred UX"),
]


def classify(value: float, bands: list[tuple[float, str]]) -> str:
    for threshold, label in bands:
        if value <= threshold:
            return label
    return bands[-1][1]


def build_command(config: dict, audio_container_path: str, threads: int, beam_size: int) -> list[str]:
    if config["engine"] == "faster-whisper":
        return [
            "/usr/bin/time",
            "-v",
            "python",
            "/bench/transcribe.py",
            "--audio",
            audio_container_path,
            "--model",
            config.get("model", "small"),
            "--threads",
            str(threads),
            "--beam-size",
            str(beam_size),
            "--language",
            "tr",
            "--batch-size",
            str(config.get("batch_size", 1)),
        ]
    return [
        "/usr/bin/time",
        "-v",
        "python",
        "/bench/transcribe.py",
        "--audio",
        audio_container_path,
        "--model",
        f"{WHISPER_CPP_MODEL_DIR}/ggml-{config['model']}.bin",
        "--threads",
        str(threads),
        "--beam-size",
        str(beam_size),
        "--language",
        "tr",
    ]


def run_once(
    config: dict,
    audio_container_path: str,
    threads: int,
    beam_size: int,
    run_index: int,
    kind: str,
) -> dict:
    if config["engine"] == "faster-whisper":
        image = FASTER_WHISPER_IMAGE
        volumes = [
            f"{DATA_DIR}:/audio:ro",
            f"{CACHE_DIR / 'huggingface'}:/cache/huggingface",
        ]
        environment = {"OMP_NUM_THREADS": str(threads), "HF_HOME": "/cache/huggingface"}
    else:
        image = WHISPER_CPP_IMAGE
        volumes = [f"{DATA_DIR}:/audio:ro", f"{CACHE_DIR / 'whisper-cpp'}:{WHISPER_CPP_MODEL_DIR}"]
        environment = {"OMP_NUM_THREADS": str(threads)}

    log(f"    {kind} run {run_index}: {config['id']}")
    command = build_command(config, audio_container_path, threads, beam_size)
    completed = run_in_container(image, command, volumes=volumes, environment=environment)
    peak_rss_kb, wall_seconds = parse_time_output(completed.stderr)
    payload = parse_engine_json(completed.stdout)
    payload["peak_rss_kb"] = peak_rss_kb
    payload["peak_rss_mb"] = round(peak_rss_kb / 1024, 1) if peak_rss_kb else None
    payload["wall_seconds"] = wall_seconds
    payload["run"] = run_index
    payload["kind"] = kind
    return payload


def summarize(runs: list[dict], audio_seconds: float, reference: str) -> dict:
    sys.path.insert(0, str(STT_DIR))
    from normalize import score  # benchmark-local module

    inference = [run["inference_seconds"] for run in runs if run["inference_seconds"]]
    if len(inference) != len(runs):
        raise RuntimeError("missing inference time in at least one run")
    wall_values = [run["wall_seconds"] for run in runs if run["wall_seconds"]]
    rss_values = [run["peak_rss_kb"] for run in runs if run["peak_rss_kb"]]
    load_values = [
        run["model_load_seconds"] for run in runs if run.get("model_load_seconds") is not None
    ]

    median_seconds = statistics.median(inference)
    median_run = min(runs, key=lambda run: abs(run["inference_seconds"] - median_seconds))
    quality = score(reference, median_run["transcript"])
    normalized = {score(reference, run["transcript"])["normalized_hypothesis"] for run in runs}

    return {
        "seconds": {
            "median": round(median_seconds, 3),
            "fastest": round(min(inference), 3),
            "slowest": round(max(inference), 3),
        },
        "wall_seconds_median": round(statistics.median(wall_values), 3) if wall_values else None,
        "rtf_median": round(median_seconds / audio_seconds, 4),
        "peak_rss_mb": round(max(rss_values) / 1024, 1) if rss_values else None,
        "model_load_seconds_median": round(statistics.median(load_values), 3)
        if load_values
        else None,
        "transcripts_identical_across_runs": len(normalized) == 1,
        "quality": {
            "wer": round(quality["wer"], 4),
            "substitutions": quality["substitutions"],
            "deletions": quality["deletions"],
            "insertions": quality["insertions"],
            "errors": quality["errors"],
            "reference_words": quality["reference_words"],
            "hypothesis_words": quality["hypothesis_words"],
        },
        "transcript": median_run["transcript"],
    }


def extrapolate(rtf: float, minutes: int) -> float:
    return minutes * 60 * rtf / 60


def markdown_small(results: dict) -> str:
    samples = results["samples"]
    sample_keys = list(samples.keys())
    rows = []
    for config in results["configs"]:
        per_sample = config["samples"]
        near = per_sample["sample_near"]["summary"]
        far = per_sample["sample_far"]["summary"]
        rows.append(
            "| {label} | {nw:.4f} | {fw:.4f} | {nr:.4f} | {fr:.4f} | {rss} MB |".format(
                label=config["label"],
                nw=near["quality"]["wer"],
                fw=far["quality"]["wer"],
                nr=near["rtf_median"],
                fr=far["rtf_median"],
                rss=config["peak_rss_mb"],
            )
        )

    lines = [
        "# CPU STT benchmark matrix",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "| Configuration | Near WER | Far WER | Near RTF | Far RTF | Peak RSS |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Medians across both recordings",
        "",
        "WER across both recordings uses total errors / total reference words (no averaging).",
        "",
        "| Configuration | Median WER | Median RTF | Total errors | Total ref words |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in results["configs"]:
        combined = config["combined"]
        lines.append(
            "| {label} | {wer:.4f} | {rtf:.4f} | {errors} | {words} |".format(
                label=config["label"],
                wer=combined["wer"],
                rtf=combined["rtf_median"],
                errors=combined["total_errors"],
                words=combined["total_reference_words"],
            )
        )

    lines += [
        "",
        "## Audio conditions",
        "",
        "| Sample | Duration | Mean volume | Max volume | Clipping | Size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in sample_keys:
        sample = samples[key]
        lines.append(
            "| {key} | {duration:.2f} s | {mean} dB | {max} dB | {clip} | {size} B |".format(
                key=key,
                duration=sample["duration_seconds"],
                mean=sample["mean_volume_db"],
                max=sample["max_volume_db"],
                clip="yes" if sample["clipping"] else "no",
                size=sample["size_bytes"],
            )
        )

    lines += [
        "",
        "## Extrapolated processing time (median RTF across both recordings)",
        "",
        "> Linear extrapolation only — ignores diarization, I/O and overheads.",
        "",
        "| Configuration | 30 min | 60 min | 120 min |",
        "|---|---:|---:|---:|",
    ]
    for config in results["configs"]:
        rtf = config["combined"]["rtf_median"]
        lines.append(
            f"| {config['label']} | {extrapolate(rtf, 30):.2f} min | "
            f"{extrapolate(rtf, 60):.2f} min | {extrapolate(rtf, 120):.2f} min |"
        )

    lines += [
        "",
        "## Raw transcripts (median run per configuration/sample, not normalized)",
        "",
    ]
    for config in results["configs"]:
        for key in sample_keys:
            lines += [
                f"### {config['label']} — {key}",
                "",
                f"```text\n{config['samples'][key]['summary']['transcript']}\n```",
                "",
            ]

    lines += [
        "## Pareto-relevant metric labels (no decision taken)",
        "",
        results["pareto"]["fastest"],
        "",
        results["pareto"]["lowest_peak_rss"],
        "",
        results["pareto"]["lowest_wer_near"],
        "",
        results["pareto"]["lowest_wer_far"],
        "",
        results["pareto"]["best_speed_quality_compromise"],
        "",
    ]
    return "\n".join(lines)


def load_small_baselines() -> list[dict]:
    """Small-tier baselines from the previous round (results/matrix.json)."""
    matrix_path = RESULTS_DIR / TIERS["small"]["json"]
    if matrix_path.exists():
        previous = json.loads(matrix_path.read_text(encoding="utf-8"))
        rows = []
        for config in previous.get("configs", []):
            if config["id"] not in {"fw-small-int8-b1", "wc-small-q8_0"}:
                continue
            rows.append(
                {
                    "label": config["label"],
                    "wer": config["combined"]["wer"],
                    "rtf": config["combined"]["rtf_median"],
                    "rss_mb": config["peak_rss_mb"],
                }
            )
        if rows:
            return rows
    return SMALL_BASELINES


def markdown_turbo(results: dict, metadata: dict | None) -> str:
    samples = results["samples"]
    sample_keys = list(samples.keys())

    lines = [
        "# CPU STT benchmark — large-v3-turbo tier",
        "",
        f"Generated: {results['generated_at']}",
        "",
        "| Configuration | Near WER | Far WER | Combined WER | Near RTF | Far RTF | Median RTF | Peak RSS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in results["configs"]:
        near = config["samples"]["sample_near"]["summary"]
        far = config["samples"]["sample_far"]["summary"]
        lines.append(
            "| {label} | {nw:.4f} | {fw:.4f} | {cw:.4f} | {nr:.4f} | {fr:.4f} | {mr:.4f} | {rss} MB |".format(
                label=config["label"],
                nw=near["quality"]["wer"],
                fw=far["quality"]["wer"],
                cw=config["combined"]["wer"],
                nr=near["rtf_median"],
                fr=far["rtf_median"],
                mr=config["combined"]["rtf_median"],
                rss=config["peak_rss_mb"],
            )
        )

    lines += [
        "",
        "## With previous small-tier baselines",
        "",
        "| Configuration | Combined WER | Median RTF | Peak RSS |",
        "|---|---:|---:|---:|",
    ]
    for baseline in load_small_baselines():
        lines.append(
            f"| {baseline['label']} | {baseline['wer']:.4f} | {baseline['rtf']:.4f} | "
            f"{baseline['rss_mb']} MB |"
        )
    for config in results["configs"]:
        lines.append(
            f"| {config['label']} | {config['combined']['wer']:.4f} | "
            f"{config['combined']['rtf_median']:.4f} | {config['peak_rss_mb']} MB |"
        )

    lines += [
        "",
        "## Project evaluation bands (thresholds, not adjusted after the fact)",
        "",
        "| Configuration | Combined WER | WER band | Median RTF | RTF band |",
        "|---|---:|---|---:|---|",
    ]
    for config in results["configs"]:
        wer = config["combined"]["wer"]
        rtf = config["combined"]["rtf_median"]
        lines.append(
            f"| {config['label']} | {wer:.4f} | {classify(wer, WER_BANDS)} | {rtf:.4f} | "
            f"{classify(rtf, RTF_BANDS)} |"
        )

    lines += [
        "",
        "## Model load time (excluded from inference time)",
        "",
        "| Configuration | Median model load |",
        "|---|---:|",
    ]
    for config in results["configs"]:
        loads = [
            config["samples"][key]["summary"]["model_load_seconds_median"] for key in sample_keys
        ]
        loads = [value for value in loads if value is not None]
        median_load = f"{statistics.median(loads):.3f} s" if loads else "not reported by engine"
        lines.append(f"| {config['label']} | {median_load} |")

    lines += [
        "",
        "## Audio conditions",
        "",
        "| Sample | Duration | Mean volume | Max volume | Clipping | Size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in sample_keys:
        sample = samples[key]
        lines.append(
            "| {key} | {duration:.2f} s | {mean} dB | {max} dB | {clip} | {size} B |".format(
                key=key,
                duration=sample["duration_seconds"],
                mean=sample["mean_volume_db"],
                max=sample["max_volume_db"],
                clip="yes" if sample["clipping"] else "no",
                size=sample["size_bytes"],
            )
        )

    lines += [
        "",
        "## Extrapolated processing time (median RTF, labeled extrapolation)",
        "",
        "| Configuration | 30 min | 60 min | 120 min |",
        "|---|---:|---:|---:|",
    ]
    for config in results["configs"]:
        rtf = config["combined"]["rtf_median"]
        lines.append(
            f"| {config['label']} | {extrapolate(rtf, 30):.2f} min | "
            f"{extrapolate(rtf, 60):.2f} min | {extrapolate(rtf, 120):.2f} min |"
        )

    lines += [
        "",
        "## Resource safety",
        "",
        "| Configuration | Peak RSS | Swap observed | Model disk size |",
        "|---|---:|---|---:|",
    ]
    for config in results["configs"]:
        disk = "n/a"
        if metadata:
            if config["engine"] == "faster-whisper":
                disk = f"{metadata['faster_whisper']['cache_size_bytes']} B (cache)"
            else:
                disk = f"{metadata['whisper_cpp']['size_bytes']} B"
        lines.append(
            f"| {config['label']} | {config['peak_rss_mb']} MB | "
            f"{'yes' if config['swap_observed'] else 'no'} | {disk} |"
        )

    lines += ["", "## Raw transcripts (median run per configuration/sample)", ""]
    for config in results["configs"]:
        for key in sample_keys:
            lines += [
                f"### {config['label']} — {key}",
                "",
                f"```text\n{config['samples'][key]['summary']['transcript']}\n```",
                "",
            ]

    lines += ["## Metric labels (no decision taken)", ""]
    for label in results["pareto"].values():
        lines.append(f"- {label}")
    lines.append("")
    return "\n".join(lines)


def markdown_report(results: dict, tier: str) -> str:
    if tier == "turbo":
        metadata_path = RESULTS_DIR / TIERS["turbo"]["metadata"]
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else None
        )
        return markdown_turbo(results, metadata)
    return markdown_small(results)


def pareto_labels(configs: list[dict]) -> dict[str, str]:
    def label(config: dict) -> str:
        return config["label"]

    fastest = min(configs, key=lambda c: c["combined"]["rtf_median"])
    lowest_rss = min(configs, key=lambda c: c["peak_rss_mb"] or float("inf"))
    lowest_wer_near = min(configs, key=lambda c: c["samples"]["sample_near"]["summary"]["quality"]["wer"])
    lowest_wer_far = min(configs, key=lambda c: c["samples"]["sample_far"]["summary"]["quality"]["wer"])
    compromise = min(
        configs,
        key=lambda c: c["combined"]["rtf_median"] * c["combined"]["wer"],
    )
    return {
        "fastest": f"fastest: {label(fastest)} (median RTF {fastest['combined']['rtf_median']:.4f})",
        "lowest_peak_rss": f"lowest peak RSS: {label(lowest_rss)} ({lowest_rss['peak_rss_mb']} MB)",
        "lowest_wer_near": f"lowest WER on sample_near: {label(lowest_wer_near)} "
        f"(WER {lowest_wer_near['samples']['sample_near']['summary']['quality']['wer']:.4f})",
        "lowest_wer_far": f"lowest WER on sample_far: {label(lowest_wer_far)} "
        f"(WER {lowest_wer_far['samples']['sample_far']['summary']['quality']['wer']:.4f})",
        "best_speed_quality_compromise": f"best RTF x WER product (lower is better): {label(compromise)} "
        f"(RTF {compromise['combined']['rtf_median']:.4f}, WER {compromise['combined']['wer']:.4f})",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--far-audio", type=Path, default=None)
    parser.add_argument("--near-audio", type=Path, default=None)
    parser.add_argument("--reference", type=Path, default=STT_DIR / "reference_tr.txt")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tier", choices=["small", "turbo"], default="small")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args()

    tier = TIERS[args.tier]
    log(f"tier: {args.tier} ({len(tier['configs'])} configurations)")

    samples = resolve_samples(args.far_audio, args.near_audio)
    for key, path in samples.items():
        log(f"{key}: {path} ({wav_duration_seconds(path):.2f} s)")

    if not args.skip_setup:
        build_images()
        if args.tier == "turbo":
            download_turbo_models()
        else:
            download_models()
            prepare_q8_0_model()
    if args.setup:
        log("setup complete")
        return 0

    reference = args.reference.read_text(encoding="utf-8")
    conditions = {key: audio_conditions(path) for key, path in samples.items()}
    for key, condition in conditions.items():
        log(
            f"{key}: {condition['duration_seconds']:.2f} s | mean {condition['mean_volume_db']} dB | "
            f"max {condition['max_volume_db']} dB | clipping: {condition['clipping']}"
        )

    container_arch = docker(
        ["run", "--rm", "--entrypoint", "uname", FASTER_WHISPER_IMAGE, "-m"]
    ).stdout.strip()
    container_nproc = int(
        docker(["run", "--rm", "--entrypoint", "nproc", FASTER_WHISPER_IMAGE]).stdout.strip()
    )
    if args.threads > container_nproc:
        raise SystemExit(f"requested {args.threads} threads but the container exposes {container_nproc}")

    runs_dir = RESULTS_DIR / tier["runs"]
    runs_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    memory_before = vm_memory_kb(FASTER_WHISPER_IMAGE)
    log(
        f"container memory: {memory_before.get('MemTotal')} kB total, "
        f"swap free {memory_before.get('SwapFree')} kB"
    )

    config_results: list[dict] = []
    for config in tier["configs"]:
        log(f"config {config['id']} ({config['label']})")
        per_sample: dict[str, dict] = {}
        peak_rss_values: list[float] = []
        swap_free_min = memory_before.get("SwapFree")

        for key, path in samples.items():
            audio_container_path = container_audio_path(path)
            audio_seconds = wav_duration_seconds(path)

            log(f"  {key}: warmup ({args.warmup} run(s), excluded)")
            for index in range(1, args.warmup + 1):
                run_once(config, audio_container_path, args.threads, args.beam_size, index, "warmup")

            measured: list[dict] = []
            for index in range(1, args.runs + 1):
                payload = run_once(
                    config, audio_container_path, args.threads, args.beam_size, index, "measured"
                )
                measured.append(payload)
                (runs_dir / f"{config['id']}-{key}-run{index}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            summary = summarize(measured, audio_seconds, reference)
            per_sample[key] = {"runs": measured, "summary": summary}
            peak_rss_values.append(summary["peak_rss_mb"] or 0.0)
            swap_now = vm_memory_kb(FASTER_WHISPER_IMAGE).get("SwapFree")
            if swap_now is not None:
                swap_free_min = min(swap_free_min or swap_now, swap_now)
            log(
                f"    -> median {summary['seconds']['median']:.3f} s | RTF {summary['rtf_median']:.4f} | "
                f"WER {summary['quality']['wer']:.4f} | peak RSS {summary['peak_rss_mb']} MB"
            )

        total_errors = sum(
            per_sample[key]["summary"]["quality"]["errors"] for key in samples
        )
        total_reference_words = sum(
            per_sample[key]["summary"]["quality"]["reference_words"] for key in samples
        )
        config_results.append(
            {
                "id": config["id"],
                "label": config["label"],
                "engine": config["engine"],
                "settings": {
                    "batch_size": config.get("batch_size"),
                    "model": config.get("model", "small"),
                    "mode": per_sample["sample_near"]["runs"][0].get("mode"),
                },
                "peak_rss_mb": max(peak_rss_values),
                "swap_free_min_kb": swap_free_min,
                "swap_observed": bool(
                    swap_free_min is not None
                    and memory_before.get("SwapFree") is not None
                    and swap_free_min < memory_before["SwapFree"]
                ),
                "samples": per_sample,
                "combined": {
                    "wer": round(total_errors / total_reference_words, 4)
                    if total_reference_words
                    else float("nan"),
                    "total_errors": total_errors,
                    "total_reference_words": total_reference_words,
                    "rtf_median": round(
                        statistics.median(
                            [per_sample[key]["summary"]["rtf_median"] for key in samples]
                        ),
                        4,
                    ),
                },
            }
        )

    results = {
        "tier": args.tier,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "samples": conditions,
        "environment": {
            "host_arch": host_arch(),
            "container_arch": container_arch,
            "nproc": container_nproc,
            "threads": args.threads,
            "beam_size": args.beam_size,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "vad": False,
            "host_os": f"{platform.system()} {platform.release()}",
            "docker": docker(["version", "--format", "{{.Server.Version}}"]).stdout.strip(),
        },
        "configs": config_results,
    }
    results["pareto"] = pareto_labels(config_results)

    json_path = RESULTS_DIR / tier["json"]
    markdown_path = RESULTS_DIR / tier["markdown"]
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(markdown_report(results, args.tier), encoding="utf-8")
    log(f"wrote {json_path}")
    log(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
