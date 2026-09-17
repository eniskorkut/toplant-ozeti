#!/usr/bin/env python3
"""CPU speaker diarization benchmark (official sherpa-onnx models only).

Runs inside the existing backend image (sherpa-onnx 1.10.46 is already installed
there); no new Python dependency is added to the application.

Configurations:
    D1  pyannote segmentation 3.0 + 3D-Speaker ERes2Net base embedding
    D2  pyannote segmentation 3.0 + NeMo TitaNet Small embedding

Modes:
    known   num_clusters = 3 (control run: the true speaker count is known)
    auto    num_clusters = -1 with a threshold sweep (0.50 / 0.65 / 0.80 / 0.90)

Usage:
    python3 benchmarks/diarization/run_benchmark.py --setup
    python3 benchmarks/diarization/run_benchmark.py --skip-setup --audio data/meetings/<id>/processing.wav
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

DIAR_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIAR_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "meetings"
MODELS_DIR = DIAR_DIR / "models"
RESULTS_DIR = DIAR_DIR / "results"
RUNS_DIR = RESULTS_DIR / "runs"

BACKEND_IMAGE = "meeting-intelligence-backend:dev"

SEGMENTATION_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models"
EMBEDDING_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models"

SEGMENTATION_ARCHIVE = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"

EMBEDDINGS = {
    "3d-speaker": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "titanet": "nemo_en_titanet_small.onnx",
}

CONFIGS = [
    {
        "id": "d1-3d-speaker",
        "label": "pyannote 3.0 + 3D-Speaker ERes2Net base",
        "embedding": "3d-speaker",
    },
    {
        "id": "d2-titanet",
        "label": "pyannote 3.0 + NeMo TitaNet Small",
        "embedding": "titanet",
    },
]

THRESHOLD_SWEEP = [0.50, 0.65, 0.80, 0.90]

# Optional approximate turn order supplied by the operator after recording
# (one label per turn, e.g. "A B A B ..."). Used for evaluation only; anonymous
# clusters are mapped to those labels with the best permutation AFTER inference.
REFERENCE_TURNS_FILE = DIAR_DIR / "reference_turns.txt"
SHORT_TURN_SECONDS = 1.0

TEST_LABEL = "two-speaker playback-through-microphone test"
DISCLAIMER = (
    "Not equivalent to a real physical multi-speaker meeting benchmark: the audio is a "
    "two-speaker conversation played back through a device and captured by the browser "
    "microphone pipeline."
)


def log(message: str) -> None:
    print(message, flush=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl(url: str, destination: Path) -> None:
    subprocess.run(
        ["curl", "-fsSL", url, "-o", str(destination)],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def official_checksums() -> dict[str, str]:
    output = subprocess.run(
        ["curl", "-fsSL", f"{EMBEDDING_RELEASE}/checksum.txt"],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    ).stdout
    checksums: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2:
            checksums[parts[0]] = parts[1]
    return checksums


def setup_models() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    archive = MODELS_DIR / SEGMENTATION_ARCHIVE

    if not archive.exists():
        log(f"downloading official segmentation model: {SEGMENTATION_ARCHIVE}")
        curl(f"{SEGMENTATION_RELEASE}/{SEGMENTATION_ARCHIVE}", archive)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(MODELS_DIR)

    embedding_checksums = official_checksums()
    for embedding in EMBEDDINGS.values():
        target = MODELS_DIR / embedding
        if target.exists():
            continue
        log(f"downloading official embedding model: {embedding}")
        curl(f"{EMBEDDING_RELEASE}/{embedding}", target)

    metadata: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segmentation": {},
        "embeddings": {},
        "note": "official k2-fsa/sherpa-onnx release assets; no third-party conversions",
    }

    segmentation_path = MODELS_DIR / SEGMENTATION_MODEL
    if not segmentation_path.exists():
        raise SystemExit(f"segmentation model missing after extraction: {segmentation_path}")
    metadata["segmentation"] = {
        "model": SEGMENTATION_MODEL,
        "archive": SEGMENTATION_ARCHIVE,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_of(archive),
        "model_size_bytes": segmentation_path.stat().st_size,
        "model_sha256": sha256_of(segmentation_path),
        "official_checksum_available": False,
    }

    for key, filename in EMBEDDINGS.items():
        path = MODELS_DIR / filename
        local_sha = sha256_of(path)
        expected = embedding_checksums.get(filename)
        if expected is None:
            raise SystemExit(f"no official checksum published for {filename}")
        if expected != local_sha:
            raise SystemExit(
                f"checksum mismatch for {filename}: expected {expected}, got {local_sha}"
            )
        metadata["embeddings"][key] = {
            "model": filename,
            "size_bytes": path.stat().st_size,
            "sha256": local_sha,
            "official_sha256": expected,
            "verified": True,
            "int8_variant": "not published by the official project; FP32 used",
        }
        log(f"verified {filename} against the official checksum")

    (RESULTS_DIR / "model-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def newest_recording() -> Path:
    candidates = sorted(
        DATA_DIR.glob("*/processing.wav"), key=lambda path: path.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise SystemExit("no recording found; record the multi-speaker script first")
    return candidates[0].resolve()


def container_audio_path(audio: Path) -> str:
    return f"/audio/{audio.relative_to(DATA_DIR.resolve()).as_posix()}"


def run_single(
    *,
    audio: Path,
    embedding: str,
    num_clusters: int,
    threshold: float,
    label: str,
    threads: int,
    min_duration_on: float,
    min_duration_off: float,
) -> dict:
    command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{MODELS_DIR}:/models:ro",
        "-v",
        f"{DATA_DIR}:/audio:ro",
        "-v",
        f"{DIAR_DIR}:/diarization:ro",
        BACKEND_IMAGE,
        "python",
        "/diarization/diarize.py",
        "--audio",
        container_audio_path(audio),
        "--segmentation-model",
        f"/models/{SEGMENTATION_MODEL}",
        "--embedding-model",
        f"/models/{EMBEDDINGS[embedding]}",
        "--threads",
        str(threads),
        "--num-clusters",
        str(num_clusters),
        "--threshold",
        str(threshold),
        "--min-duration-on",
        str(min_duration_on),
        "--min-duration-off",
        str(min_duration_off),
        "--label",
        label,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise SystemExit(
            f"diarization run failed ({label}):\n{completed.stderr[-3000:]}"
        )
    for line in reversed(completed.stdout.strip().splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return json.loads(candidate)
    raise SystemExit(f"no JSON payload in output:\n{completed.stdout[-2000:]}")


def load_reference_turns(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    tokens = [token for token in path.read_text(encoding="utf-8").replace(",", " ").split() if token]
    return tokens or None


def detected_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker segments into turns."""
    turns: list[dict] = []
    for segment in sorted(segments, key=lambda item: item["start"]):
        if turns and turns[-1]["speaker"] == segment["speaker"]:
            turns[-1]["end"] = segment["end"]
        else:
            turns.append({"speaker": segment["speaker"], "start": segment["start"], "end": segment["end"]})
    return turns


def align_sequences(mapped: list[str], expected: list[str]) -> tuple[int, list[dict]]:
    """Global alignment (match +1, mismatch/gap -1) returning score and per-expected outcome."""
    n, m = len(expected), len(mapped)
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] - 1
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] - 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = score[i - 1][j - 1] + (1 if expected[i - 1] == mapped[j - 1] else -1)
            score[i][j] = max(match, score[i - 1][j] - 1, score[i][j - 1] - 1)

    i, j = n, m
    outcome = [
        {"expected": expected[index], "detected": None, "matched": False}
        for index in range(n)
    ]
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            match = score[i - 1][j - 1] + (1 if expected[i - 1] == mapped[j - 1] else -1)
            if score[i][j] == match:
                outcome[i - 1] = {
                    "expected": expected[i - 1],
                    "detected": mapped[j - 1],
                    "matched": expected[i - 1] == mapped[j - 1],
                }
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] - 1:
            i -= 1
            continue
        j -= 1

    return score[n][m], outcome


def evaluate(raw: dict, reference: list[str] | None) -> dict:
    """Structural metrics always; turn-sequence metrics only with a reference order."""
    segments = raw["segments"]
    turns = detected_turns(segments)
    cluster_sequence = [turn["speaker"] for turn in turns]

    durations: dict[str, float] = {}
    for segment in segments:
        key = str(segment["speaker"])
        durations[key] = durations.get(key, 0.0) + (segment["end"] - segment["start"])
    total_speech = sum(durations.values()) or 1.0

    short_turns_detected = [
        {
            "speaker": str(turn["speaker"]),
            "start": round(turn["start"], 3),
            "duration": round(turn["end"] - turn["start"], 3),
        }
        for turn in turns
        if (turn["end"] - turn["start"]) < SHORT_TURN_SECONDS
    ]

    result: dict = {
        "num_speakers": raw["num_speakers"],
        "num_segments": raw["num_segments"],
        "num_turns_after_merge": len(turns),
        "speaker_durations_seconds": {key: round(value, 2) for key, value in sorted(durations.items())},
        "speaker_duration_share": {
            key: round(value / total_speech, 3) for key, value in sorted(durations.items())
        },
        "short_turns_detected": short_turns_detected,
        "reference_turn_order_provided": reference is not None,
    }

    if reference is None:
        result.update(
            {
                "turn_sequence": None,
                "cluster_to_label_mapping": None,
                "turn_matches": None,
                "turn_total": None,
                "turn_consistency": None,
                "missed_reference_turns": None,
                "merged_errors": None,
                "split_errors": None,
            }
        )
        return result

    cluster_ids = sorted({str(cluster) for cluster in cluster_sequence})
    labels = sorted(set(reference))
    mapping_candidates = min(len(labels), len(cluster_ids))
    best: dict | None = None
    for permutation in permutations(labels, mapping_candidates):
        mapping = {cluster: permutation[index] for index, cluster in enumerate(cluster_ids[:mapping_candidates])}
        mapped = [mapping.get(str(cluster), "X") for cluster in cluster_sequence]
        score, outcome = align_sequences(mapped, reference)
        if best is None or score > best["score"]:
            best = {"score": score, "mapping": mapping, "mapped": mapped, "outcome": outcome}

    assert best is not None
    matched = [item for item in best["outcome"] if item["matched"]]
    missed_reference_turns = sum(1 for item in best["outcome"] if item["detected"] is None)

    occurrences: dict[str, int] = {}
    for item in best["outcome"]:
        if item["detected"]:
            occurrences[item["detected"]] = occurrences.get(item["detected"], 0) + 1
    expected_counts: dict[str, int] = {}
    for label in reference:
        expected_counts[label] = expected_counts.get(label, 0) + 1
    split_errors = sum(
        max(0, count - expected_counts.get(label, 0)) for label, count in occurrences.items()
    )
    merged_errors = sum(
        1
        for index, item in enumerate(best["outcome"])
        if index > 0 and item["detected"] and item["detected"] == best["outcome"][index - 1]["detected"]
    )

    result.update(
        {
            "turn_sequence": best["mapped"],
            "cluster_to_label_mapping": {str(key): value for key, value in best["mapping"].items()},
            "turn_matches": len(matched),
            "turn_total": len(reference),
            "turn_consistency": round(len(matched) / len(reference), 4),
            "missed_reference_turns": missed_reference_turns,
            "merged_errors": merged_errors,
            "split_errors": split_errors,
        }
    )
    return result


def summarize(runs: list[dict]) -> dict:
    seconds = [run["inference_seconds"] for run in runs]
    rtf = [run["rtf"] for run in runs]
    rss = [run["peak_rss_kb"] for run in runs]
    return {
        "inference_seconds_median": round(statistics.median(seconds), 3),
        "inference_seconds_fastest": round(min(seconds), 3),
        "inference_seconds_slowest": round(max(seconds), 3),
        "rtf_median": round(statistics.median(rtf), 4),
        "peak_rss_mb": round(max(rss) / 1024, 1),
        "model_load_seconds_median": round(
            statistics.median([run["model_load_seconds"] for run in runs]), 3
        ),
        "num_speakers_all_runs": [run["num_speakers"] for run in runs],
        "num_speakers_stable": len({run["num_speakers"] for run in runs}) == 1,
    }


def markdown_report(results: dict) -> str:
    known = results["known_count"]
    auto = results["auto_count"]
    mode_label = results["known_count_mode_label"]

    def percent(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "n/a"

    lines = [
        "# CPU speaker diarization benchmark (sherpa-onnx 1.10.46)",
        "",
        f"**Test type: {results['test_label']}**",
        "",
        f"> {results['disclaimer']}",
        "",
        f"Generated: {results['generated_at']}",
        "",
        f"Audio: `{results['audio']['path']}` ({results['audio']['duration_seconds']:.2f} s, "
        f"mean {results['audio']['mean_volume_db']} dB, max {results['audio']['max_volume_db']} dB)",
        "",
        f"Environment: {results['environment']['container_arch']}, threads "
        f"{results['environment']['threads']}, CPU only, 1 warmup + "
        f"{results['environment']['measured_runs']} measured runs per known-count config.",
        "",
        f"Expected speakers: {results['expected_speakers']}. Reference turn order: "
        + (
            f"{results['reference_turn_count']} turns provided by the operator."
            if results["reference_turn_order_provided"]
            else "not provided (sequence metrics are n/a)."
        ),
        "",
        "## Result matrix",
        "",
        "| Config | Count mode | Detected speakers | Consistency | Missed short turns | RTF | Peak RSS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for config in results["configs"]:
        known_run = known[config["id"]]
        auto_run = auto[config["id"]]
        known_eval = known_run["evaluation"]
        auto_run_record = next(
            run for run in auto_run["runs"] if run["threshold"] == auto_run["best_stable_threshold"]
        )
        auto_eval = auto_run_record["evaluation"]
        missed_known = (
            str(known_eval["missed_reference_turns"])
            if known_eval["missed_reference_turns"] is not None
            else f"n/a ({len(known_eval['short_turns_detected'])} detected <1 s)"
        )
        missed_auto = (
            str(auto_eval["missed_reference_turns"])
            if auto_eval["missed_reference_turns"] is not None
            else f"n/a ({len(auto_eval['short_turns_detected'])} detected <1 s)"
        )
        lines.append(
            "| {label} | {mode} | {speakers} | {consistency} | {missed} | {rtf:.4f} | {rss} MB |".format(
                label=config["label"],
                mode=mode_label,
                speakers=known_run["summary"]["num_speakers_all_runs"][0],
                consistency=percent(known_eval["turn_consistency"]),
                missed=missed_known,
                rtf=known_run["summary"]["rtf_median"],
                rss=known_run["summary"]["peak_rss_mb"],
            )
        )
        lines.append(
            "| {label} | auto (stable count {count}, threshold {threshold}) | {speakers} | {consistency} | {missed} | {rtf:.4f} | {rss} MB |".format(
                label=config["label"],
                count=auto_run["stable_speaker_count"],
                threshold=auto_run["best_stable_threshold"],
                speakers=auto_eval["num_speakers"],
                consistency=percent(auto_eval["turn_consistency"]),
                missed=missed_auto,
                rtf=auto_run_record["rtf"],
                rss=auto_run_record["peak_rss_mb"],
            )
        )

    lines += [
        "",
        "## Automatic clustering threshold sweep (single run per threshold)",
        "",
        "| Config | Threshold | Detected speakers | Segments | Turns | Short turns <1 s | Duration share |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for config in results["configs"]:
        for run in auto[config["id"]]["runs"]:
            evaluation = run["evaluation"]
            share = ", ".join(
                f"{key}: {value:.0%}" for key, value in evaluation["speaker_duration_share"].items()
            )
            lines.append(
                f"| {config['label']} | {run['threshold']:.2f} | {run['num_speakers']} | "
                f"{run['num_segments']} | {evaluation['num_turns_after_merge']} | "
                f"{len(evaluation['short_turns_detected'])} | {share} |"
            )

    lines += [
        "",
        f"## Speaker separation detail ({mode_label}, median run)",
        "",
        "| Config | Mapped turn sequence detected | Matches | Missed | Merged | Split |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in results["configs"]:
        evaluation = known[config["id"]]["evaluation"]
        sequence = (
            " ".join(evaluation["turn_sequence"]) if evaluation["turn_sequence"] else "n/a"
        )
        lines.append(
            f"| {config['label']} | {sequence} | "
            f"{evaluation['turn_matches'] if evaluation['turn_matches'] is not None else 'n/a'}"
            f"/{evaluation['turn_total'] if evaluation['turn_total'] is not None else 'n/a'} | "
            f"{evaluation['missed_reference_turns'] if evaluation['missed_reference_turns'] is not None else 'n/a'} | "
            f"{evaluation['merged_errors'] if evaluation['merged_errors'] is not None else 'n/a'} | "
            f"{evaluation['split_errors'] if evaluation['split_errors'] is not None else 'n/a'} |"
        )

    lines += [
        "",
        "## Performance",
        "",
        "| Config | Mode | Model load (median) | Inference (median) | RTF (median) | Peak RSS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in results["configs"]:
        for mode, payload in ((mode_label, known[config["id"]]), ("auto", auto[config["id"]])):
            summary = payload.get("summary") or {
                "model_load_seconds_median": payload["runs"][0]["model_load_seconds"],
                "inference_seconds_median": payload["runs"][0]["inference_seconds"],
                "rtf_median": payload["runs"][0]["rtf"],
                "peak_rss_mb": payload["runs"][0]["peak_rss_mb"],
            }
            lines.append(
                f"| {config['label']} | {mode} | {summary['model_load_seconds_median']} s | "
                f"{summary['inference_seconds_median']} s | {summary['rtf_median']:.4f} | "
                f"{summary['peak_rss_mb']} MB |"
            )

    lines += [
        "",
        "## Extrapolated processing time (median RTF; labeled extrapolation)",
        "",
        "| Config | Mode | 30 min | 60 min | 120 min |",
        "|---|---|---:|---:|---:|",
    ]
    for config in results["configs"]:
        for mode, payload in ((mode_label, known[config["id"]]), ("auto", auto[config["id"]])):
            rtf = (payload.get("summary") or {"rtf_median": payload["runs"][0]["rtf"]})["rtf_median"]
            lines.append(
                f"| {config['label']} | {mode} | {30 * 60 * rtf / 60:.2f} min | "
                f"{60 * 60 * rtf / 60:.2f} min | {120 * 60 * rtf / 60:.2f} min |"
            )

    lines += ["", "## Metric labels (no production decision taken)", ""]
    for label in results["labels"].values():
        lines.append(f"- {label}")
    lines.append("")
    return "\n".join(lines)


def metric_labels(results: dict) -> dict[str, str]:
    entries = [
        (config["label"], results["known_count"][config["id"]], results["auto_count"][config["id"]])
        for config in results["configs"]
    ]

    def median_rtf(entry) -> float:
        payload = entry[1].get("summary") or {"rtf_median": entry[1]["runs"][0]["rtf"]}
        return payload["rtf_median"]

    def rss(entry) -> float:
        payload = entry[1].get("summary") or {"peak_rss_mb": entry[1]["runs"][0]["peak_rss_mb"]}
        return payload["peak_rss_mb"]

    fastest = min(entries, key=median_rtf)
    lowest_ram = min(entries, key=rss)
    most_stable = min(
        entries, key=lambda e: len(set(e[1]["summary"]["num_speakers_all_runs"]))
    )

    stability_sets = {
        tuple(e[1]["summary"]["num_speakers_all_runs"]) for e in entries
    }
    if len(stability_sets) == 1:
        stable_counts = sorted({count for counts in stability_sets for count in counts})
        stability_label = (
            f"most stable speaker count ({results['known_count_mode_label']} mode): tie — all "
            f"configurations returned {stable_counts} speakers in every measured run"
        )
    else:
        stability_label = (
            f"most stable speaker count ({results['known_count_mode_label']} mode): {most_stable[0]}"
        )
    labels = {
        "fastest": f"fastest: {fastest[0]} (median RTF {median_rtf(fastest):.4f})",
        "lowest_ram": f"lowest peak RSS: {lowest_ram[0]} ({rss(lowest_ram)} MB)",
        "most_stable_count": stability_label,
    }

    if results["reference_turn_order_provided"]:
        by_consistency = [entry for entry in entries if entry[1]["evaluation"]["turn_consistency"] is not None]
        if by_consistency:
            best = max(by_consistency, key=lambda e: e[1]["evaluation"]["turn_consistency"])
            labels["best_consistency"] = (
                f"highest turn consistency: {best[0]} "
                f"({best[1]['evaluation']['turn_consistency']:.2%})"
            )
            fewest_missed = min(by_consistency, key=lambda e: e[1]["evaluation"]["missed_reference_turns"])
            labels["fewest_missed_turns"] = (
                f"fewest missed reference turns: {fewest_missed[0]} "
                f"({fewest_missed[1]['evaluation']['missed_reference_turns']} missed)"
            )
    else:
        labels["consistency"] = (
            "turn-consistency metrics: n/a (no reference turn order was provided for this recording)"
        )

    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--expected-speakers", type=int, default=2)
    parser.add_argument("--reference-turns", type=Path, default=REFERENCE_TURNS_FILE)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-duration-on", type=float, default=0.3)
    parser.add_argument("--min-duration-off", type=float, default=0.5)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--skip-setup", action="store_true")
    args = parser.parse_args()

    if not args.skip_setup:
        setup_models()
    if args.setup:
        log("setup complete")
        return 0

    audio = (args.audio.resolve() if args.audio else newest_recording())
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DIAR_DIR))
    from audio_conditions import measure  # benchmark-local helper

    reference_turns = load_reference_turns(args.reference_turns)
    if reference_turns is None:
        log(f"no reference turn order at {args.reference_turns}; sequence metrics will be n/a")
    else:
        log(f"reference turn order loaded: {len(reference_turns)} turns")

    condition = measure(audio)
    log(
        f"audio: {audio} | {condition['duration_seconds']:.2f} s | mean {condition['mean_volume_db']} dB | "
        f"max {condition['max_volume_db']} dB"
    )

    known_count: dict[str, dict] = {}
    for config in CONFIGS:
        log(
            f"{config['id']}: known-count mode (num_clusters={args.expected_speakers}), "
            f"warmup + {args.runs} measured"
        )
        for index in range(1, args.warmup + 1):
            run_single(
                audio=audio,
                embedding=config["embedding"],
                num_clusters=args.expected_speakers,
                threshold=0.5,
                label=f"{config['id']}-known-warmup{index}",
                threads=args.threads,
                min_duration_on=args.min_duration_on,
                min_duration_off=args.min_duration_off,
            )

        measured: list[dict] = []
        for index in range(1, args.runs + 1):
            raw = run_single(
                audio=audio,
                embedding=config["embedding"],
                num_clusters=args.expected_speakers,
                threshold=0.5,
                label=f"{config['id']}-known-run{index}",
                threads=args.threads,
                min_duration_on=args.min_duration_on,
                min_duration_off=args.min_duration_off,
            )
            raw["rtf"] = round(raw["inference_seconds"] / raw["audio_seconds"], 4)
            raw["peak_rss_mb"] = round(raw["peak_rss_kb"] / 1024, 1)
            raw["evaluation"] = evaluate(raw, reference_turns)
            measured.append(raw)
            (RUNS_DIR / f"{config['id']}-known-run{index}.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            consistency = raw["evaluation"]["turn_consistency"]
            consistency_text = f"{consistency:.2%}" if consistency is not None else "n/a"
            log(
                f"    -> {raw['inference_seconds']:.3f} s | RTF {raw['rtf']:.4f} | "
                f"speakers {raw['num_speakers']} | segments {raw['num_segments']} | "
                f"consistency {consistency_text} | peak RSS {raw['peak_rss_mb']} MB"
            )

        median_run = sorted(measured, key=lambda run: run["inference_seconds"])[len(measured) // 2]
        known_count[config["id"]] = {
            "runs": measured,
            "summary": summarize(measured),
            "evaluation": median_run["evaluation"],
        }

    auto_count: dict[str, dict] = {}
    for config in CONFIGS:
        log(f"{config['id']}: automatic clustering sweep {THRESHOLD_SWEEP}")
        sweep_runs: list[dict] = []
        for threshold in THRESHOLD_SWEEP:
            raw = run_single(
                audio=audio,
                embedding=config["embedding"],
                num_clusters=-1,
                threshold=threshold,
                label=f"{config['id']}-auto-{threshold}",
                threads=args.threads,
                min_duration_on=args.min_duration_on,
                min_duration_off=args.min_duration_off,
            )
            raw["rtf"] = round(raw["inference_seconds"] / raw["audio_seconds"], 4)
            raw["peak_rss_mb"] = round(raw["peak_rss_kb"] / 1024, 1)
            raw["evaluation"] = evaluate(raw, reference_turns)
            sweep_runs.append(raw)
            (RUNS_DIR / f"{config['id']}-auto-{threshold}.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            consistency = raw["evaluation"]["turn_consistency"]
            consistency_text = f"{consistency:.2%}" if consistency is not None else "n/a"
            log(
                f"    threshold {threshold:.2f} -> speakers {raw['num_speakers']} | "
                f"segments {raw['num_segments']} | consistency {consistency_text}"
            )

        # Stability is judged by agreement across neighbouring thresholds, not by
        # a single lucky value: pick the most frequent speaker count.
        counts = [run["num_speakers"] for run in sweep_runs]
        stable_count = max(set(counts), key=counts.count)
        stable_runs = [run for run in sweep_runs if run["num_speakers"] == stable_count]
        if reference_turns is not None:
            best = max(stable_runs, key=lambda run: run["evaluation"]["turn_consistency"] or 0.0)
        else:
            best = stable_runs[0]
        auto_count[config["id"]] = {
            "runs": sweep_runs,
            "stable_speaker_count": stable_count,
            "best_stable_threshold": best["threshold"],
        }

    results = {
        "test_label": TEST_LABEL,
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio": condition,
        "environment": {
            "container_arch": subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", "uname", BACKEND_IMAGE, "-m"],
                capture_output=True,
                text=True,
                check=True,
                stdin=subprocess.DEVNULL,
            ).stdout.strip(),
            "host_os": f"{platform.system()} {platform.release()}",
            "threads": args.threads,
            "sherpa_onnx": "1.10.46",
            "window_shift_ratio": "not present in sherpa-onnx 1.10.46 (segmentation window shift is internal)",
            "min_duration_on": args.min_duration_on,
            "min_duration_off": args.min_duration_off,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
        },
        "configs": CONFIGS,
        "known_count": known_count,
        "auto_count": auto_count,
        "known_count_mode_label": f"known={args.expected_speakers}",
        "expected_speakers": args.expected_speakers,
        "reference_turn_order_provided": reference_turns is not None,
        "reference_turn_count": len(reference_turns) if reference_turns else None,
    }
    results["labels"] = metric_labels(results)

    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(markdown_report(results), encoding="utf-8")
    log(f"wrote {RESULTS_DIR / 'latest.json'}")
    log(f"wrote {RESULTS_DIR / 'latest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
