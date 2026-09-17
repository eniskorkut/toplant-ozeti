#!/usr/bin/env python3
"""Reproducible CPU STT benchmark: faster-whisper vs whisper.cpp.

Runs entirely in isolated Docker images (no STT dependency touches the
application runtime). Standard library only on the host.

Examples:
    python3 benchmarks/stt/run_benchmark.py --setup
    python3 benchmarks/stt/run_benchmark.py --threads 8 --runs 3
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

CONTAINER_AUDIO_ROOT = "/audio"
WHISPER_CPP_MODEL_DIR = "/models"

MAX_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
WALL_CLOCK_RE = re.compile(r"Elapsed \(wall clock\) time .*?: (\d+):(\d+(?:\.\d+)?)")


def log(message: str) -> None:
    print(message, flush=True)


def docker(
    args: list[str], *, capture: bool = True, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=capture,
        text=True,
        check=check,
        stdin=subprocess.DEVNULL,
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


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newest_recording() -> Path:
    candidates = sorted(
        (path for path in DATA_DIR.glob("*/processing.wav")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"no recording found under {DATA_DIR}; record one in the browser first")
    return candidates[0]


def container_audio_path(audio: Path) -> str:
    relative = audio.resolve().relative_to(DATA_DIR.resolve())
    return f"{CONTAINER_AUDIO_ROOT}/{relative.as_posix()}"


def build_images() -> None:
    log(f"building {FASTER_WHISPER_IMAGE} (faster-whisper 1.2.1, python 3.12 slim)")
    docker(["build", "-t", FASTER_WHISPER_IMAGE, str(FASTER_WHISPER_DIR)])
    log(f"building {WHISPER_CPP_IMAGE} (whisper.cpp pinned release, CPU only)")
    docker(["build", "-t", WHISPER_CPP_IMAGE, str(WHISPER_CPP_DIR)])


def download_models(whisper_cpp_model: str, faster_whisper_model: str, threads: int) -> None:
    (CACHE_DIR / "huggingface").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "whisper-cpp").mkdir(parents=True, exist_ok=True)

    log(f"downloading faster-whisper model: {faster_whisper_model} (into benchmark cache)")
    run_in_container(
        FASTER_WHISPER_IMAGE,
        [
            "python",
            "-c",
            (
                "from faster_whisper import WhisperModel;"
                f"WhisperModel('{faster_whisper_model}', device='cpu', compute_type='int8',"
                f" cpu_threads={threads});print('cached')"
            ),
        ],
        volumes=[f"{CACHE_DIR / 'huggingface'}:/cache/huggingface"],
        environment={"HF_HOME": "/cache/huggingface"},
    )

    log(f"downloading whisper.cpp model: {whisper_cpp_model} (official distribution)")
    run_in_container(
        WHISPER_CPP_IMAGE,
        [
            "/opt/whisper.cpp/models/download-ggml-model.sh",
            whisper_cpp_model,
            WHISPER_CPP_MODEL_DIR,
        ],
        volumes=[f"{CACHE_DIR / 'whisper-cpp'}:{WHISPER_CPP_MODEL_DIR}"],
        environment={},
    )


def candidate_commands(
    *,
    audio_container_path: str,
    threads: int,
    beam_size: int,
    faster_whisper_model: str,
    whisper_cpp_model_path: str,
) -> dict[str, list[str]]:
    return {
        "faster-whisper": [
            "/usr/bin/time",
            "-v",
            "python",
            "/bench/transcribe.py",
            "--audio",
            audio_container_path,
            "--model",
            faster_whisper_model,
            "--threads",
            str(threads),
            "--beam-size",
            str(beam_size),
            "--language",
            "tr",
        ],
        "whisper.cpp": [
            "/usr/bin/time",
            "-v",
            "python",
            "/bench/transcribe.py",
            "--audio",
            audio_container_path,
            "--model",
            whisper_cpp_model_path,
            "--threads",
            str(threads),
            "--beam-size",
            str(beam_size),
            "--language",
            "tr",
        ],
    }


def candidate_volumes() -> dict[str, list[str]]:
    return {
        "faster-whisper": [
            f"{DATA_DIR}:/audio:ro",
            f"{CACHE_DIR / 'huggingface'}:/cache/huggingface",
        ],
        "whisper.cpp": [
            f"{DATA_DIR}:/audio:ro",
            f"{CACHE_DIR / 'whisper-cpp'}:{WHISPER_CPP_MODEL_DIR}",
        ],
    }


def run_once(
    name: str,
    command: list[str],
    volumes: list[str],
    threads: int,
    run_index: int,
    kind: str,
) -> dict:
    image = FASTER_WHISPER_IMAGE if name == "faster-whisper" else WHISPER_CPP_IMAGE
    environment = {"OMP_NUM_THREADS": str(threads)}
    if name == "faster-whisper":
        environment["HF_HOME"] = "/cache/huggingface"

    log(f"  {kind} run {run_index}: {name}")
    completed = run_in_container(image, command, volumes=volumes, environment=environment)
    peak_rss_kb, wall_seconds = parse_time_output(completed.stderr)
    payload = parse_engine_json(completed.stdout)

    payload["peak_rss_kb"] = peak_rss_kb
    payload["peak_rss_mb"] = round(peak_rss_kb / 1024, 1) if peak_rss_kb else None
    payload["wall_seconds"] = wall_seconds
    payload["run"] = run_index
    payload["kind"] = kind
    return payload


def summarize(name: str, runs: list[dict], audio_seconds: float, reference: str) -> dict:
    from normalize import score  # benchmark-local module

    inference_seconds = [run["inference_seconds"] for run in runs if run["inference_seconds"]]
    if len(inference_seconds) != len(runs):
        raise RuntimeError(f"{name}: missing inference time in at least one run")

    median_seconds = statistics.median(inference_seconds)
    median_run = min(runs, key=lambda run: abs(run["inference_seconds"] - median_seconds))
    quality = score(reference, median_run["transcript"])

    peak_rss = [run["peak_rss_kb"] for run in runs if run["peak_rss_kb"]]
    wall_seconds_values = [run["wall_seconds"] for run in runs if run["wall_seconds"]]
    normalized_transcripts = {score(reference, run["transcript"])["normalized_hypothesis"] for run in runs}

    return {
        "runs": len(runs),
        "seconds": {
            "median": round(median_seconds, 3),
            "fastest": round(min(inference_seconds), 3),
            "slowest": round(max(inference_seconds), 3),
        },
        "wall_seconds": {
            "median": round(statistics.median(wall_seconds_values), 3)
            if wall_seconds_values
            else None,
        },
        "rtf": {
            "median": round(median_seconds / audio_seconds, 4),
            "from_wall_clock": round(
                statistics.median(wall_seconds_values) / audio_seconds, 4
            )
            if wall_seconds_values
            else None,
        },
        "peak_rss_mb": round(max(peak_rss) / 1024, 1) if peak_rss else None,
        "transcripts_identical_across_runs": len(normalized_transcripts) == 1,
        "model_load_seconds": median_run.get("model_load_seconds"),
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


def markdown_report(results: dict) -> str:
    faster = results["candidates"]["faster-whisper"]
    whisper = results["candidates"]["whisper.cpp"]
    audio_seconds = results["audio"]["duration_seconds"]

    def row(label: str, left: object, right: object) -> str:
        return f"| {label} | {left} | {right} |"

    lines = [
        "# CPU STT benchmark — faster-whisper vs whisper.cpp",
        "",
        f"Generated: {results['generated_at']}",
        "",
        f"Audio: `{results['audio']['path']}` ({audio_seconds:.2f} s, sha256 "
        f"`{results['audio']['sha256'][:16]}…`)",
        "",
        f"Environment: host {results['environment']['host_arch']}, container "
        f"{results['environment']['container_arch']}, nproc {results['environment']['nproc']}, "
        f"threads {results['environment']['threads']}, warmup runs per candidate: "
        f"{results['environment']['warmup_runs']} (excluded), measured runs: "
        f"{results['environment']['measured_runs']}, VAD off, model caches persisted outside "
        "the measured runs.",
        "",
        "| Metric | faster-whisper | whisper.cpp |",
        "|---|---:|---:|",
        row("model", faster["config"]["model"], whisper["config"]["model"]),
        row("precision/quant", faster["config"]["compute_type"], whisper["config"]["quantization"]),
        row("threads", faster["config"]["threads"], whisper["config"]["threads"]),
        row(
            "median time",
            f"{faster['summary']['seconds']['median']:.3f} s",
            f"{whisper['summary']['seconds']['median']:.3f} s",
        ),
        row(
            "median RTF",
            f"{faster['summary']['rtf']['median']:.4f}",
            f"{whisper['summary']['rtf']['median']:.4f}",
        ),
        row(
            "peak RAM",
            f"{faster['summary']['peak_rss_mb']} MB",
            f"{whisper['summary']['peak_rss_mb']} MB",
        ),
        row("WER", f"{faster['summary']['quality']['wer']:.4f}", f"{whisper['summary']['quality']['wer']:.4f}"),
        row(
            "substitutions",
            faster["summary"]["quality"]["substitutions"],
            whisper["summary"]["quality"]["substitutions"],
        ),
        row(
            "deletions",
            faster["summary"]["quality"]["deletions"],
            whisper["summary"]["quality"]["deletions"],
        ),
        row(
            "insertions",
            faster["summary"]["quality"]["insertions"],
            whisper["summary"]["quality"]["insertions"],
        ),
        "",
        "## Extrapolated processing time (based on median RTF only)",
        "",
        "> These are linear extrapolations from the median RTF measured on a "
        f"{audio_seconds:.2f} s recording. They ignore diarization, I/O and overheads; treat them "
        "as estimates, not promises.",
        "",
        "| Audio length | faster-whisper | whisper.cpp |",
        "|---|---:|---:|",
    ]

    for label, minutes in (("30 minute meeting", 30), ("60 minute meeting", 60), ("120 minute meeting", 120)):
        seconds = minutes * 60
        lines.append(
            row(
                label,
                f"{seconds * faster['summary']['rtf']['median'] / 60:.2f} min",
                f"{seconds * whisper['summary']['rtf']['median'] / 60:.2f} min",
            )
        )

    lines += [
        "",
        "## Raw transcripts (median run, not normalized)",
        "",
        "### faster-whisper",
        "",
        f"```text\n{faster['summary']['transcript']}\n```",
        "",
        "### whisper.cpp",
        "",
        f"```text\n{whisper['summary']['transcript']}\n```",
        "",
        "## Qualitative observations (manual, kept separate from metrics)",
        "",
        "(to be filled in after reviewing the raw transcripts)",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=None, help="processing.wav to benchmark")
    parser.add_argument(
        "--reference",
        type=Path,
        default=STT_DIR / "reference_tr.txt",
        help="reference transcript for WER",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--faster-whisper-model", default="small")
    parser.add_argument("--whisper-cpp-model", default="small-q5_1")
    parser.add_argument("--setup", action="store_true", help="build images and download models only")
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(STT_DIR))

    audio = (args.audio or newest_recording()).resolve()
    reference = args.reference.read_text(encoding="utf-8")
    audio_seconds = wav_duration_seconds(audio)
    audio_container_path = container_audio_path(audio)
    whisper_cpp_model_path = f"{WHISPER_CPP_MODEL_DIR}/ggml-{args.whisper_cpp_model}.bin"

    log(f"audio: {audio}")
    log(f"duration: {audio_seconds:.3f} s | threads: {args.threads} | beam: {args.beam_size}")

    if not args.skip_setup:
        build_images()
        download_models(args.whisper_cpp_model, args.faster_whisper_model, args.threads)
    if args.setup:
        log("setup complete")
        return 0

    container_arch = docker(
        ["run", "--rm", "--entrypoint", "uname", FASTER_WHISPER_IMAGE, "-m"]
    ).stdout.strip()
    container_nproc = int(
        docker(["run", "--rm", "--entrypoint", "nproc", FASTER_WHISPER_IMAGE]).stdout.strip()
    )
    if args.threads > container_nproc:
        raise SystemExit(
            f"requested {args.threads} threads but the container exposes {container_nproc}"
        )

    commands = candidate_commands(
        audio_container_path=audio_container_path,
        threads=args.threads,
        beam_size=args.beam_size,
        faster_whisper_model=args.faster_whisper_model,
        whisper_cpp_model_path=whisper_cpp_model_path,
    )
    volumes = candidate_volumes()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    candidates: dict[str, dict] = {}
    for name, command in commands.items():
        log(f"{name}: warmup ({args.warmup} run(s), excluded from statistics)")
        for index in range(1, args.warmup + 1):
            run_once(name, command, volumes[name], args.threads, index, "warmup")

        measured: list[dict] = []
        for index in range(1, args.runs + 1):
            payload = run_once(name, command, volumes[name], args.threads, index, "measured")
            measured.append(payload)
            (RUNS_DIR / f"{name}-run{index}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        summary = summarize(name, measured, audio_seconds, reference)
        candidates[name] = {
            "config": {
                "model": args.faster_whisper_model if name == "faster-whisper" else whisper_cpp_model_path,
                "compute_type": measured[0].get("compute_type"),
                "quantization": "ggml q5_1" if name == "whisper.cpp" else None,
                "threads": args.threads,
                "beam_size": args.beam_size,
                "vad": False,
                "language": "tr",
                "engine_version": measured[0].get("engine_version"),
            },
            "runs": measured,
            "summary": summary,
        }
        log(
            f"  -> median {summary['seconds']['median']:.3f} s | RTF "
            f"{summary['rtf']['median']:.4f} | peak RSS {summary['peak_rss_mb']} MB | "
            f"WER {summary['quality']['wer']:.4f}"
        )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio": {
            "path": str(audio.relative_to(REPO_ROOT)),
            "duration_seconds": round(audio_seconds, 3),
            "size_bytes": audio.stat().st_size,
            "sha256": sha256_of(audio),
        },
        "environment": {
            "host_arch": host_arch(),
            "container_arch": container_arch,
            "nproc": container_nproc,
            "threads": args.threads,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "host_os": f"{platform.system()} {platform.release()}",
            "docker": docker(["version", "--format", "{{.Server.Version}}"]).stdout.strip(),
        },
        "reference_words": len(reference.split()),
        "candidates": candidates,
    }

    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(markdown_report(results), encoding="utf-8")
    log(f"wrote {RESULTS_DIR / 'latest.json'}")
    log(f"wrote {RESULTS_DIR / 'latest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
