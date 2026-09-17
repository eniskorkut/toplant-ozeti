#!/usr/bin/env python3
"""VoxConverse 0.3 diarization benchmark (ground-truth RTTM scoring).

Runs in stages so long benchmarks can be resumed:

    --mode setup       build the pinned dscore image, extract the 12 selected WAVs
    --mode known       known-count control (num_clusters = reference speaker count)
    --mode calibrate   automatic clustering, threshold sweep on the calibration split
    --mode validate    selected threshold, held-out validation split
    --mode repeat      validation set repeated 3x with the selected threshold
    --mode report      write results/voxconverse.json + .md

Scoring uses the pinned dscore checkout (collar 0.25, overlaps included as primary;
overlaps ignored as a diagnostic). Per-file missed speech / false alarm / speaker
error are not exposed by this dscore version and are reported as unavailable.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DIAR_DIR = Path(__file__).resolve().parent
DATASET_DIR = DIAR_DIR / "datasets" / "voxconverse"
RTTM_DIR = DATASET_DIR / "voxconverse" / "dev"
ARCHIVE = DATASET_DIR / "voxconverse_dev_wav.zip"
AUDIO_DIR = DATASET_DIR / "audio"
MODELS_DIR = DIAR_DIR / "models"
RESULTS_DIR = DIAR_DIR / "results"
SCORING_DIR = RESULTS_DIR / "scoring"
STATE_FILE = RESULTS_DIR / "voxconverse-state.json"

BACKEND_IMAGE = "meeting-intelligence-backend:dev"
DSCORE_IMAGE = "mi-bench-dscore:local"
DSCORE_DOCKERFILE = DIAR_DIR / "dscore.Dockerfile"

SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDINGS = {
    "d1-3d-speaker": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "d2-titanet": "nemo_en_titanet_small.onnx",
}
CONFIG_LABELS = {
    "d1-3d-speaker": "3D-Speaker",
    "d2-titanet": "TitaNet",
}

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
COLLAR = 0.25
REPEATS = 3


METADATA_EMBEDDING_KEYS = {"d1-3d-speaker": "3d-speaker", "d2-titanet": "titanet"}


def log(message: str) -> None:
    print(message, flush=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"runs": {}, "scoring": {}, "selection": {}, "meta": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def subset_entries() -> list[dict]:
    metadata = DATASET_DIR / "subset-metadata.json"
    if not metadata.exists():
        raise SystemExit("subset metadata missing; run select_voxconverse_subset.py first")
    entries = json.loads(metadata.read_text(encoding="utf-8"))
    for entry in entries:
        entry["reference_speakers"] = entry["reference"]["num_speakers"]
    order = {"two": 0, "three": 1, "four_plus": 2}
    entries.sort(key=lambda item: (order[item["group"]], item["duration_seconds"], item["file_id"]))

    seen: dict[str, int] = {}
    for entry in entries:
        seen[entry["group"]] = seen.get(entry["group"], 0) + 1
        entry["split"] = "calibration" if seen[entry["group"]] <= 2 else "validation"
    return entries


def reference_rttm(file_id: str) -> Path:
    path = RTTM_DIR / f"{file_id}.rttm"
    if not path.exists():
        raise SystemExit(f"reference RTTM missing: {path}")
    return path


def audio_path(file_id: str) -> Path:
    return AUDIO_DIR / f"{file_id}.wav"


def ensure_setup() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    entries = subset_entries()
    missing = [entry for entry in entries if not audio_path(entry["file_id"]).exists()]
    if missing:
        with zipfile.ZipFile(ARCHIVE) as archive:
            members = {Path(name).stem: name for name in archive.namelist() if name.endswith(".wav")}
            for entry in missing:
                member = members[entry["file_id"]]
                with archive.open(member) as source, audio_path(entry["file_id"]).open("wb") as target:
                    target.write(source.read())
                log(f"extracted {member}")

    image_exists = subprocess.run(
        ["docker", "image", "inspect", DSCORE_IMAGE],
        capture_output=True,
        check=False,
        stdin=subprocess.DEVNULL,
    ).returncode == 0
    if not image_exists:
        log(f"building {DSCORE_IMAGE} (dscore pinned commit)")
        subprocess.run(
            ["docker", "build", "-f", str(DSCORE_DOCKERFILE), "-t", DSCORE_IMAGE, str(DIAR_DIR)],
            check=True,
            stdin=subprocess.DEVNULL,
        )


def run_diarization(
    *,
    file_id: str,
    embedding: str,
    num_clusters: int,
    threshold: float,
    label: str,
) -> dict:
    sys_rttm = SCORING_DIR / label / f"{file_id}.rttm"
    sys_rttm.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "docker", "run", "--rm",
        "-v", f"{MODELS_DIR}:/models:ro",
        "-v", f"{AUDIO_DIR}:/audio:ro",
        "-v", f"{DIAR_DIR}:/diarization:ro",
        "-v", f"{SCORING_DIR}:/diarization/results/scoring:rw",
        BACKEND_IMAGE,
        "python", "/diarization/diarize.py",
        "--audio", f"/audio/{file_id}.wav",
        "--segmentation-model", f"/models/{SEGMENTATION_MODEL}",
        "--embedding-model", f"/models/{EMBEDDINGS[embedding]}",
        "--threads", "8",
        "--num-clusters", str(num_clusters),
        "--threshold", str(threshold),
        "--min-duration-on", "0.3",
        "--min-duration-off", "0.5",
        "--label", label,
        "--rttm-out", f"/diarization/results/scoring/{label}/{file_id}.rttm",
        "--rttm-file-id", file_id,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise SystemExit(f"diarization failed ({label}):\n{completed.stderr[-2500:]}")
    for line in reversed(completed.stdout.strip().splitlines()):
        candidate = line.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return json.loads(candidate)
    raise SystemExit(f"no JSON payload for {label}")


def score_group(system_rttms: dict[str, Path], *, ignore_overlaps: bool) -> dict:
    reference_paths = [str(reference_rttm(file_id)) for file_id in system_rttms]
    system_paths = [str(path) for path in system_rttms.values()]
    command = [
        "docker", "run", "--rm",
        "-e", "PYTHONPATH=/opt/dscore",
        "-v", f"{DATASET_DIR}:/dataset:ro",
        "-v", f"{DIAR_DIR}:/diarization:ro",
        DSCORE_IMAGE,
        "python", "/diarization/score_rttm.py",
        "--reference", *[path.replace(str(DATASET_DIR), "/dataset") for path in reference_paths],
        "--system", *[path.replace(str(DIAR_DIR), "/diarization") for path in system_paths],
        "--collar", str(COLLAR),
    ]
    if ignore_overlaps:
        command.append("--ignore-overlaps")
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise SystemExit(f"scoring failed:\n{completed.stderr[-2500:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def record_group(
    state: dict,
    *,
    entries: list[dict],
    embedding: str,
    mode: str,
    threshold: float | None,
    state_key: str,
    repeat: int = 0,
) -> dict:
    group_runs: list[dict] = []
    for entry in entries:
        num_clusters = entry["reference_speakers"] if mode == "known" else -1
        label = f"{state_key}--{entry['file_id']}"
        started = time.perf_counter()
        raw = run_diarization(
            file_id=entry["file_id"],
            embedding=embedding,
            num_clusters=num_clusters,
            threshold=threshold if threshold is not None else 0.5,
            label=label,
        )
        elapsed = time.perf_counter() - started
        run = {
            "file_id": entry["file_id"],
            "group": entry["group"],
            "split": entry["split"],
            "reference_speakers": entry["reference_speakers"],
            "audio_seconds": raw["audio_seconds"],
            "model": embedding,
            "mode": mode,
            "threshold": threshold,
            "repeat": repeat,
            "detected_speakers": raw["num_speakers"],
            "segments": raw["num_segments"],
            "inference_seconds": raw["inference_seconds"],
            "model_load_seconds": raw["model_load_seconds"],
            "peak_rss_mb": round(raw["peak_rss_kb"] / 1024, 1),
            "rtf": round(raw["inference_seconds"] / raw["audio_seconds"], 4),
            "wall_seconds": round(elapsed, 3),
            "sys_rttm": f"scoring/{label}/{entry['file_id']}.rttm",
        }
        group_runs.append(run)
        log(
            f"    {entry['file_id']:<7} {entry['reference_speakers']} spk -> "
            f"{raw['num_speakers']} | {raw['inference_seconds']:.2f} s | RTF {run['rtf']:.4f} | "
            f"{run['peak_rss_mb']} MB"
        )

    system_rttms = {
        run["file_id"]: DIAR_DIR / "results" / run["sys_rttm"] for run in group_runs
    }
    primary = score_group(system_rttms, ignore_overlaps=False)
    diagnostic = score_group(system_rttms, ignore_overlaps=True)

    detected = [run["detected_speakers"] for run in group_runs]
    reference = [run["reference_speakers"] for run in group_runs]
    exact = sum(1 for detected_value, reference_value in zip(detected, reference) if detected_value == reference_value)
    mae = sum(abs(d - r) for d, r in zip(detected, reference)) / len(group_runs)

    payload = {
        "embedding": embedding,
        "mode": mode,
        "threshold": threshold,
        "repeat": repeat,
        "files": [run["file_id"] for run in group_runs],
        "runs": group_runs,
        "der_overlap_included": primary["global"]["der"],
        "jer": primary["global"]["jer"],
        "der_overlap_ignored": diagnostic["global"]["der"],
        "per_file_der": primary["per_file"],
        "count_exact": exact,
        "count_total": len(group_runs),
        "count_accuracy": round(exact / len(group_runs), 4),
        "count_mae": round(mae, 4),
        "unavailable_metrics": primary["unavailable_metrics"],
    }
    state["scoring"][state_key] = payload
    save_state(state)
    return payload


def select_thresholds(state: dict, entries: list[dict]) -> None:
    calibration = [entry for entry in entries if entry["split"] == "calibration"]
    for embedding in EMBEDDINGS:
        candidates = []
        for threshold in THRESHOLDS:
            key = f"calibrate--{embedding}--{threshold:.2f}"
            payload = state["scoring"].get(key)
            if payload is None:
                continue
            candidates.append((threshold, payload))
        if not candidates:
            continue
        # Values are compared at 1e-6 precision: differences below that are
        # floating-point noise, not a metric difference, and must fall through
        # to the documented tie breakers.
        def rank(item: tuple[float, dict]) -> tuple:
            threshold, payload = item
            return (
                round(payload["der_overlap_included"], 6),
                round(payload["jer"], 6),
                round(payload["count_mae"], 6),
                abs(threshold - 0.50),
            )

        candidates.sort(key=rank)
        threshold, payload = candidates[0]
        state["selection"][embedding] = {
            "threshold": threshold,
            "calibration_der": payload["der_overlap_included"],
            "calibration_jer": payload["jer"],
            "calibration_count_accuracy": payload["count_accuracy"],
            "calibration_count_mae": payload["count_mae"],
        }
        log(
            f"selected threshold for {embedding}: {threshold:.2f} "
            f"(calibration DER {payload['der_overlap_included']:.2f}%, JER {payload['jer']:.2f}%)"
        )
    state["meta"]["calibration_files"] = [entry["file_id"] for entry in calibration]
    save_state(state)


def validation_entries(state: dict, entries: list[dict]) -> list[dict]:
    return [entry for entry in entries if entry["split"] == "validation"]


def summarize_runs(runs: list[dict]) -> dict:
    return {
        "median_rtf": round(statistics.median([run["rtf"] for run in runs]), 4),
        "median_inference_seconds": round(statistics.median([run["inference_seconds"] for run in runs]), 3),
        "median_model_load_seconds": round(
            statistics.median([run["model_load_seconds"] for run in runs]), 3
        ),
        "peak_rss_mb": round(max(run["peak_rss_mb"] for run in runs), 1),
        "total_audio_seconds": round(sum(run["audio_seconds"] for run in runs), 2),
    }


def build_report(state: dict, entries: list[dict]) -> tuple[dict, str]:
    selection = state["selection"]
    known_keys = {embedding: f"known--{embedding}--all" for embedding in EMBEDDINGS}

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "VoxConverse",
            "version": "0.3 (annotations: joonson/voxconverse master)",
            "license": "CC BY 4.0 (research purposes)",
            "subset": len(entries),
            "audio_seconds_total": round(sum(entry["duration_seconds"] for entry in entries), 1),
        },
        "method": {
            "collar_seconds": COLLAR,
            "der_primary": "DER, collar 0.25 s, overlapping speech included",
            "der_diagnostic": "DER, collar 0.25 s, overlapping speech ignored",
            "scoring_tool": "nryant/dscore @ e02f949ac6592279300a2c33d03daf9e0c12fd27",
            "min_duration_on": 0.3,
            "min_duration_off": 0.5,
            "threads": 8,
            "known_count_control": True,
        },
        "selection": selection,
        "calibration": {
            f"{embedding}--{threshold:.2f}": state["scoring"].get(f"calibrate--{embedding}--{threshold:.2f}")
            for embedding in EMBEDDINGS
            for threshold in THRESHOLDS
            if state["scoring"].get(f"calibrate--{embedding}--{threshold:.2f}") is not None
        },
        "known_count": {embedding: state["scoring"][key] for embedding, key in known_keys.items()},
        "validation": {},
        "repeatability": {},
    }

    for embedding in EMBEDDINGS:
        key = f"validate--{embedding}--{selection[embedding]['threshold']:.2f}"
        payload = state["scoring"].get(key)
        if payload is None:
            continue
        report["validation"][embedding] = {
            **{k: payload[k] for k in (
                "der_overlap_included",
                "jer",
                "der_overlap_ignored",
                "count_accuracy",
                "count_mae",
                "per_file_der",
                "runs",
            )},
            "summary": summarize_runs(payload["runs"]),
            "by_group": {},
        }
        for group in ("two", "three", "four_plus"):
            runs = [run for run in payload["runs"] if run["group"] == group]
            if not runs:
                continue
            group_runs = [run for run in runs]
            per_file = {
                entry["file_id"]: entry["der"]
                for entry in payload["per_file_der"]
                if entry["file_id"] in {run["file_id"] for run in group_runs}
            }
            report["validation"][embedding]["by_group"][group] = {
                "files": [run["file_id"] for run in group_runs],
                "per_file_der": per_file,
                "count_accuracy": round(
                    sum(1 for run in group_runs if run["detected_speakers"] == run["reference_speakers"])
                    / len(group_runs),
                    4,
                ),
                "count_mae": round(
                    sum(abs(run["detected_speakers"] - run["reference_speakers"]) for run in group_runs)
                    / len(group_runs),
                    4,
                ),
                "median_rtf": round(statistics.median([run["rtf"] for run in group_runs]), 4),
            }

        repeat_keys = [
            f"repeat{index}--{embedding}--{selection[embedding]['threshold']:.2f}"
            for index in range(1, REPEATS + 1)
        ]
        repeat_payloads = [state["scoring"].get(key) for key in repeat_keys]
        repeat_payloads = [payload for payload in repeat_payloads if payload is not None]
        if repeat_payloads:
            report["repeatability"][embedding] = {
                "der_values": [payload["der_overlap_included"] for payload in repeat_payloads],
                "count_sequences": [
                    [run["detected_speakers"] for run in payload["runs"]] for payload in repeat_payloads
                ],
                "deterministic_count": len({tuple(run["detected_speakers"] for run in payload["runs"]) for payload in repeat_payloads}) == 1,
                "deterministic_der": len({round(payload["der_overlap_included"], 6) for payload in repeat_payloads}) == 1,
            }

    return report, markdown_report(report)


def markdown_report(report: dict) -> str:
    lines = [
        "# VoxConverse 0.3 diarization benchmark (sherpa-onnx 1.10.46)",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Dataset: {report['dataset']['name']} {report['dataset']['version']} — "
        f"{report['dataset']['subset']} recordings, "
        f"{report['dataset']['audio_seconds_total']:.1f} s total audio.",
        "",
        f"Scoring: {report['method']['scoring_tool']}, collar {report['method']['collar_seconds']} s. "
        "Primary metric includes overlapping speech; the diagnostic one ignores overlaps. "
        f"Per-file missed speech / false alarm / speaker error are unavailable in this dscore version.",
        "",
        "## Known speaker count (control)",
        "",
        "| Config | Mode | Threshold | DER | JER | Count accuracy | Count MAE | RTF | Peak RSS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for embedding, payload in report["known_count"].items():
        summary = summarize_runs(payload["runs"])
        lines.append(
            f"| {CONFIG_LABELS[embedding]} | known count | n/a | "
            f"{payload['der_overlap_included']:.2f}% | {payload['jer']:.2f}% | 100% forced | 0 forced | "
            f"{summary['median_rtf']:.4f} | {summary['peak_rss_mb']} MB |"
        )

    lines += [
        "",
        "## Automatic speaker count (held-out validation)",
        "",
        "| Config | Mode | Threshold | DER | JER | Count accuracy | Count MAE | RTF | Peak RSS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for embedding, payload in report["validation"].items():
        summary = payload["summary"]
        lines.append(
            f"| {CONFIG_LABELS[embedding]} | automatic validation | "
            f"{report['selection'][embedding]['threshold']:.2f} | {payload['der_overlap_included']:.2f}% | "
            f"{payload['jer']:.2f}% | {payload['count_accuracy']:.2%} | {payload['count_mae']:.3f} | "
            f"{summary['median_rtf']:.4f} | {summary['peak_rss_mb']} MB |"
        )

    lines += [
        "",
        "### Validation breakdown by reference speaker count",
        "",
        "| Config | Group | Files | Count accuracy | Count MAE | Median RTF | Per-file DER |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for embedding, payload in report["validation"].items():
        for group, data in payload["by_group"].items():
            per_file = ", ".join(f"{fid}: {der:.1f}%" for fid, der in data["per_file_der"].items())
            lines.append(
                f"| {CONFIG_LABELS[embedding]} | {group} | {', '.join(data['files'])} | "
                f"{data['count_accuracy']:.2%} | {data['count_mae']:.3f} | {data['median_rtf']:.4f} | {per_file} |"
            )

    lines += [
        "",
        "## Calibration threshold sweep (calibration split only, single run per threshold)",
        "",
        "| Config | Threshold | DER | JER | Count accuracy | Count MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for embedding in EMBEDDINGS:
        for threshold in THRESHOLDS:
            payload = report["calibration"].get(f"{embedding}--{threshold:.2f}")
            if payload is None:
                continue
            marker = " **<- selected**" if report["selection"].get(embedding, {}).get("threshold") == threshold else ""
            lines.append(
                f"| {CONFIG_LABELS[embedding]} | {threshold:.2f} | "
                f"{payload['der_overlap_included']:.2f}% | {payload['jer']:.2f}% | "
                f"{payload['count_accuracy']:.2%} | {payload['count_mae']:.3f}{marker} |"
            )

    lines += [
        "",
        "### Selected thresholds",
        "",
        "| Config | Threshold | Calibration DER | Calibration JER | Calibration count accuracy | Calibration count MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for embedding, data in report["selection"].items():
        lines.append(
            f"| {CONFIG_LABELS[embedding]} | {data['threshold']:.2f} | {data['calibration_der']:.2f}% | "
            f"{data['calibration_jer']:.2f}% | {data['calibration_count_accuracy']:.2%} | "
            f"{data['calibration_count_mae']:.3f} |"
        )

    lines += [
        "",
        "## Repeatability (validation set, selected threshold, 3 repetitions)",
        "",
        "| Config | DER per repetition | Deterministic speaker count | Deterministic DER |",
        "|---|---|---|---|",
    ]
    for embedding, data in report["repeatability"].items():
        lines.append(
            f"| {CONFIG_LABELS[embedding]} | "
            f"{', '.join(f'{value:.2f}%' for value in data['der_values'])} | "
            f"{'yes' if data['deterministic_count'] else 'no'} | "
            f"{'yes' if data['deterministic_der'] else 'no'} |"
        )

    metadata_path = RESULTS_DIR / "model-metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    )
    lines += [
        "",
        "## Performance and model footprint",
        "",
        "| Config | Segmentation model | Embedding model | Median model load | Median RTF | Peak RSS |",
        "|---|---|---|---:|---:|---:|",
    ]
    segmentation_size = (
        f"{metadata['segmentation']['model_size_bytes'] / 1e6:.1f} MB" if metadata else "n/a"
    )
    for embedding, payload in report["validation"].items():
        embedding_size = "n/a"
        if metadata:
            key = METADATA_EMBEDDING_KEYS[embedding]
            embedding_size = f"{metadata['embeddings'][key]['size_bytes'] / 1e6:.1f} MB"
        # run dicts are not in the report payload; reuse the known-count summary for load times
        known = report["known_count"].get(embedding)
        load_counts = [
            run["model_load_seconds"] for run in (known["runs"] if known else [])
        ]
        load_median = (
            f"{statistics.median(load_counts):.3f} s" if load_counts else "n/a"
        )
        lines.append(
            f"| {CONFIG_LABELS[embedding]} | pyannote 3.0 ({segmentation_size}) | "
            f"{embedding_size} | {load_median} | {payload['summary']['median_rtf']:.4f} | "
            f"{payload['summary']['peak_rss_mb']} MB |"
        )

    lines += [
        "",
        "## Extrapolated processing time (median validation RTF; estimates only)",
        "",
        "| Config | 30 min | 60 min | 120 min |",
        "|---|---:|---:|---:|",
    ]
    for embedding, payload in report["validation"].items():
        rtf = payload["summary"]["median_rtf"]
        lines.append(
            f"| {CONFIG_LABELS[embedding]} | {30 * 60 * rtf / 60:.2f} min | "
            f"{60 * 60 * rtf / 60:.2f} min | {120 * 60 * rtf / 60:.2f} min |"
        )

    lines += ["", "## Metric labels (no production decision taken)", ""]
    validation = report["validation"]
    if validation:
        lowest_der = min(validation.items(), key=lambda item: item[1]["der_overlap_included"])
        lowest_jer = min(validation.items(), key=lambda item: item[1]["jer"])
        best_count_value = max(item[1]["count_accuracy"] for item in validation.items())
        best_count_names = [
            CONFIG_LABELS[name]
            for name, payload in validation.items()
            if payload["count_accuracy"] == best_count_value
        ]
        lowest_mae = min(validation.items(), key=lambda item: item[1]["count_mae"])
        fastest = min(validation.items(), key=lambda item: item[1]["summary"]["median_rtf"])
        lowest_ram = min(validation.items(), key=lambda item: item[1]["summary"]["peak_rss_mb"])
        lines += [
            f"- lowest validation DER: {CONFIG_LABELS[lowest_der[0]]} "
            f"({lowest_der[1]['der_overlap_included']:.2f}%)",
            f"- lowest validation JER: {CONFIG_LABELS[lowest_jer[0]]} ({lowest_jer[1]['jer']:.2f}%)",
            f"- best automatic speaker-count accuracy: {' tie — '.join(best_count_names)} "
            f"({best_count_value:.2%})",
            f"- lowest speaker-count MAE: {CONFIG_LABELS[lowest_mae[0]]} "
            f"({lowest_mae[1]['count_mae']:.3f})",
            f"- fastest median RTF: {CONFIG_LABELS[fastest[0]]} "
            f"({fastest[1]['summary']['median_rtf']:.4f})",
            f"- lowest peak RSS: {CONFIG_LABELS[lowest_ram[0]]} "
            f"({lowest_ram[1]['summary']['peak_rss_mb']} MB)",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["setup", "known", "calibrate", "validate", "repeat", "report"],
        required=True,
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    entries = subset_entries()

    if args.mode == "setup":
        ensure_setup()
        state["meta"]["subset"] = [entry["file_id"] for entry in entries]
        save_state(state)
        log("setup complete")
        return 0

    if args.mode == "known":
        for embedding in EMBEDDINGS:
            log(f"known-count control: {embedding}")
            record_group(
                state,
                entries=entries,
                embedding=embedding,
                mode="known",
                threshold=None,
                state_key=f"known--{embedding}--all",
            )
        return 0

    if args.mode == "calibrate":
        calibration = [entry for entry in entries if entry["split"] == "calibration"]
        for embedding in EMBEDDINGS:
            for threshold in THRESHOLDS:
                key = f"calibrate--{embedding}--{threshold:.2f}"
                if key in state["scoring"]:
                    continue
                log(f"calibration: {embedding} threshold {threshold:.2f}")
                record_group(
                    state,
                    entries=calibration,
                    embedding=embedding,
                    mode="auto",
                    threshold=threshold,
                    state_key=key,
                )
        select_thresholds(state, entries)
        return 0

    if args.mode == "validate":
        if not state["selection"]:
            raise SystemExit("no threshold selected; run --mode calibrate first")
        validation = validation_entries(state, entries)
        for embedding in EMBEDDINGS:
            threshold = state["selection"][embedding]["threshold"]
            key = f"validate--{embedding}--{threshold:.2f}"
            if key in state["scoring"]:
                continue
            log(f"validation: {embedding} threshold {threshold:.2f}")
            record_group(
                state,
                entries=validation,
                embedding=embedding,
                mode="auto",
                threshold=threshold,
                state_key=key,
            )
        return 0

    if args.mode == "repeat":
        if not state["selection"]:
            raise SystemExit("no threshold selected; run --mode calibrate first")
        validation = validation_entries(state, entries)
        for index in range(1, REPEATS + 1):
            for embedding in EMBEDDINGS:
                threshold = state["selection"][embedding]["threshold"]
                key = f"repeat{index}--{embedding}--{threshold:.2f}"
                if key in state["scoring"]:
                    continue
                log(f"repeatability {index}/{REPEATS}: {embedding} threshold {threshold:.2f}")
                record_group(
                    state,
                    entries=validation,
                    embedding=embedding,
                    mode="auto",
                    threshold=threshold,
                    state_key=key,
                    repeat=index,
                )
        return 0

    if args.mode == "report":
        if not state["selection"]:
            raise SystemExit("no threshold selected; run --mode calibrate first")
        report, markdown = build_report(state, entries)
        (RESULTS_DIR / "voxconverse.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (RESULTS_DIR / "voxconverse.md").write_text(markdown, encoding="utf-8")
        log(f"wrote {RESULTS_DIR / 'voxconverse.json'}")
        log(f"wrote {RESULTS_DIR / 'voxconverse.md'}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
