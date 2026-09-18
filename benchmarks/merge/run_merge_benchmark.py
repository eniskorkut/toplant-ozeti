#!/usr/bin/env python3
"""Word-timestamp + diarization merge benchmark (benchmark-only, no app integration).

For every recording and both STT engines it produces:
    * STT words/tokens with timestamps (cached under results/raw/)
    * TitaNet diarization segments (auto threshold 0.80, plus known count where the
      reference speaker count is available)
    * a merged speaker-attributed transcript with structural diagnostics
    * for VoxConverse files, word→speaker agreement against reference RTTM turns

Nothing here touches the FastAPI application.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

MERGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MERGE_DIR.parent.parent
DIAR_DIR = REPO_ROOT / "benchmarks" / "diarization"
STT_DIR = REPO_ROOT / "benchmarks" / "stt"
DATA_DIR = REPO_ROOT / "data" / "meetings"
VOXCONVERSE_AUDIO = DIAR_DIR / "datasets" / "voxconverse" / "audio"
VOXCONVERSE_RTTM = DIAR_DIR / "datasets" / "voxconverse" / "voxconverse" / "dev"

RESULTS_DIR = MERGE_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"

BACKEND_IMAGE = "meeting-intelligence-backend:dev"
FASTER_WHISPER_IMAGE = "mi-bench-faster-whisper:local"
WHISPER_CPP_IMAGE = "mi-bench-whisper-cpp:local"

WHISPER_CPP_MODEL = "/models/ggml-large-v3-turbo-q8_0.bin"
FASTER_WHISPER_MODEL = "large-v3-turbo"
FASTER_WHISPER_CACHE = STT_DIR / ".cache" / "huggingface"
WHISPER_CPP_CACHE = STT_DIR / ".cache" / "whisper-cpp"

SEGMENTATION_MODEL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
TITANET_MODEL = "nemo_en_titanet_small.onnx"
DIAR_MODELS = DIAR_DIR / "models"

TITANET_THRESHOLD = 0.80
THREADS = 8

TURKISH_RECORDINGS = [
    {"id": "sample_far", "file_id": "fcdf1737901d41ddacd14429d39a4183", "language": "tr"},
    {"id": "sample_near", "file_id": "798b2efc586542d780eef3dce5dbbc8b", "language": "tr"},
]

VOXCONVERSE_RECORDINGS = [
    {"id": "whmpa", "file_id": "whmpa", "language": "en"},
    {"id": "jiqvr", "file_id": "jiqvr", "language": "en"},
    {"id": "jyirt", "file_id": "jyirt", "language": "en"},
    {"id": "wjhgf", "file_id": "wjhgf", "language": "en"},
]

sys.path.insert(0, str(MERGE_DIR))
from merge import DiarizationSegment, Word, merge, timestamp_health  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def reference_text() -> str:
    return (STT_DIR / "reference_tr.txt").read_text(encoding="utf-8")


def parse_rttm(path: Path) -> list[dict]:
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        turns.append({"start": start, "end": start + float(parts[4]), "speaker": parts[7]})
    return turns


def docker_run(image: str, command: list[str], volumes: list[str], environment: dict[str, str] | None = None) -> str:
    args = ["docker", "run", "--rm"]
    for volume in volumes:
        args += ["-v", volume]
    for key, value in (environment or {}).items():
        args += ["-e", f"{key}={value}"]
    args += [image, *command]
    completed = subprocess.run(args, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise SystemExit(f"command failed:\n{completed.stderr[-2500:]}")
    return completed.stdout.strip().splitlines()[-1]


def transcribe(recording: dict, engine: str, audio_container_path: str, mounts: list[str], *, force: bool) -> dict:
    raw_path = RAW_DIR / f"{recording['id']}--{engine}.json"
    if raw_path.exists() and not force:
        return json.loads(raw_path.read_text(encoding="utf-8"))

    if engine == "whisper-cpp":
        output = docker_run(
            WHISPER_CPP_IMAGE,
            [
                "python", "/bench/stt_transcribe.py",
                "--engine", "whisper-cpp",
                "--audio", audio_container_path,
                "--model", WHISPER_CPP_MODEL,
                "--language", recording["language"],
                "--threads", str(THREADS),
            ],
            [f"{MERGE_DIR}:/bench:ro", f"{WHISPER_CPP_CACHE}:/models:ro", *mounts],
        )
    else:
        output = docker_run(
            FASTER_WHISPER_IMAGE,
            [
                "python", "/bench/stt_transcribe.py",
                "--engine", "faster-whisper",
                "--audio", audio_container_path,
                "--model", FASTER_WHISPER_MODEL,
                "--language", recording["language"],
                "--threads", str(THREADS),
            ],
            [f"{MERGE_DIR}:/bench:ro", f"{FASTER_WHISPER_CACHE}:/cache/huggingface", *mounts],
            {"HF_HOME": "/cache/huggingface"},
        )
    payload = json.loads(output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def diarize(audio_container_path: str, mounts: list[str], *, num_clusters: int, label: str) -> dict:
    output = docker_run(
        BACKEND_IMAGE,
        [
            "python", "/diarization/diarize.py",
            "--audio", audio_container_path,
            "--segmentation-model", f"/models/{SEGMENTATION_MODEL}",
            "--embedding-model", f"/models/{TITANET_MODEL}",
            "--threads", str(THREADS),
            "--num-clusters", str(num_clusters),
            "--threshold", str(TITANET_THRESHOLD),
            "--min-duration-on", "0.3",
            "--min-duration-off", "0.5",
            "--label", label,
        ],
        [f"{DIAR_DIR}:/diarization:ro", f"{DIAR_MODELS}:/models:ro", *mounts],
    )
    return json.loads(output)


def reference_speaker_at(turns: list[dict], moment: float) -> str | None:
    for turn in turns:
        if turn["start"] <= moment <= turn["end"]:
            return turn["speaker"]
    return None


def assignment_agreement(assigned: list[dict], reference_turns: list[dict]) -> dict:
    """Best-permutation word→reference agreement (evaluation only, never in output)."""
    clusters = sorted({word["speaker"] for word in assigned})
    reference_speakers = sorted({turn["speaker"] for turn in reference_turns})
    limit = min(len(clusters), len(reference_speakers))

    best: dict | None = None
    for permutation in permutations(reference_speakers, limit):
        mapping = {cluster: permutation[index] for index, cluster in enumerate(clusters[:limit])}
        matches = 0
        compared = 0
        for word in assigned:
            mapped = mapping.get(word["speaker"])
            if mapped is None:
                continue
            moment = (word["start"] + word["end"]) / 2
            reference = reference_speaker_at(reference_turns, moment)
            if reference is None:
                continue
            compared += 1
            if reference == mapped:
                matches += 1
        if best is None or matches > best["matches"]:
            best = {"matches": matches, "compared": compared, "mapping": mapping}

    assert best is not None
    return {
        "clusters": clusters,
        "mapping": best["mapping"],
        "compared_words": best["compared"],
        "matching_words": best["matches"],
        "agreement": round(best["matches"] / best["compared"], 4) if best["compared"] else None,
    }


def run_recording(
    recording: dict,
    *,
    audio_path: Path,
    audio_container_path: str,
    mounts: list[str],
    reference_turns: list[dict] | None,
    reference_speakers: int | None,
    force: bool,
) -> dict:
    import wave

    with wave.open(str(audio_path), "rb") as wav_file:
        audio_seconds = wav_file.getnframes() / wav_file.getframerate()

    log(f"  diarization (TitaNet auto @ {TITANET_THRESHOLD})")
    diarization = diarize(audio_container_path, mounts, num_clusters=-1, label=f"merge-{recording['id']}-auto")
    diarization_modes = {"auto_0.80": diarization}
    if reference_speakers is not None:
        log(f"  diarization (TitaNet known count = {reference_speakers})")
        diarization_modes[f"known_{reference_speakers}"] = diarize(
            audio_container_path,
            mounts,
            num_clusters=reference_speakers,
            label=f"merge-{recording['id']}-known",
        )

    payload: dict = {
        "id": recording["id"],
        "file_id": recording["file_id"],
        "audio_seconds": round(audio_seconds, 3),
        "language": recording["language"],
        "engines": {},
    }

    for engine in ("whisper-cpp", "faster-whisper"):
        log(f"  STT: {engine}")
        stt = transcribe(recording, engine, audio_container_path, mounts, force=force)
        words = [Word(word["start"], word["end"], word["text"]) for word in stt["words"]]
        health = timestamp_health(words, audio_seconds)

        engine_result = {
            "engine": stt["engine"],
            "engine_version": stt["engine_version"],
            "timestamp_source": stt["timestamp_source"],
            "model": stt["model"],
            "words": len(words),
            "inference_seconds": stt["inference_seconds"],
            "model_load_seconds": stt["model_load_seconds"],
            "rtf": round(stt["inference_seconds"] / audio_seconds, 4),
            "peak_rss_mb": round(stt["peak_rss_kb"] / 1024, 1),
            "timestamp_health": health,
            "text": stt["text"],
            "merges": {},
        }

        if recording["language"] == "tr":
            sys.path.insert(0, str(STT_DIR))
            from normalize import score

            engine_result["wer"] = round(score(reference_text(), stt["text"])["wer"], 4)

        for mode, diarization_result in diarization_modes.items():
            segments = [
                DiarizationSegment(segment["start"], segment["end"], f"speaker_{segment['speaker']}")
                for segment in diarization_result["segments"]
            ]
            merged = merge(words, segments, audio_seconds=audio_seconds)
            if merged["structural_violations"]:
                raise SystemExit(
                    f"structural invariants violated for {recording['id']}/{engine}/{mode}: "
                    f"{merged['structural_violations'][:5]}"
                )
            entry = {
                "diarization_speakers": diarization_result["num_speakers"],
                "diarization_segments": diarization_result["num_segments"],
                **{key: merged[key] for key in (
                    "words",
                    "assigned_words",
                    "unresolved_words",
                    "speaker_changes",
                    "short_turns",
                    "rapid_flips",
                    "per_speaker_words",
                )},
                "turns": merged["turns"],
            }
            if reference_turns is not None:
                assigned_words = [
                    assignment
                    for assignment in merged["assignments"]
                    if assignment["speaker"] is not None
                ]
                entry["assignment_agreement"] = assignment_agreement(assigned_words, reference_turns)
            engine_result["merges"][mode] = entry

        payload["engines"][engine] = engine_result
    return payload


def markdown(report: dict) -> str:
    lines = [
        "# STT word timestamps + diarization merge benchmark",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Settings: sherpa-onnx 1.10.46, pyannote 3.0 + TitaNet Small, auto threshold "
        f"**{TITANET_THRESHOLD}**, {THREADS} threads, boundary tolerance "
        f"{report['settings']['boundary_tolerance_seconds']} s, short-turn threshold "
        f"{report['settings']['short_turn_seconds']} s. Known-count diarization is run "
        "wherever a reference speaker count exists.",
        "",
        "## Decision matrix",
        "",
        "| Metric | whisper.cpp | faster-whisper |",
        "|---|---:|---:|",
    ]

    def aggregate(engine: str) -> dict:
        entries = [rec["engines"][engine] for rec in report["recordings"] if engine in rec["engines"]]
        healths = [entry["timestamp_health"] for entry in entries]
        # one consistent mode for the decision matrix: the production-relevant
        # automatic speaker count (no reference count supplied to the merge)
        merges = [entry["merges"]["auto_0.80"] for entry in entries if "auto_0.80" in entry["merges"]]
        return {
            "words": sum(health["words"] for health in healths),
            "monotonicity_errors": sum(health["monotonicity_errors"] for health in healths),
            "zero_duration": sum(health["zero_duration_words"] for health in healths),
            "overlaps": sum(health["overlapping_word_timestamps"] for health in healths),
            "unresolved": sum(entry["unresolved_words"] for entry in merges),
            "assigned": sum(entry["assigned_words"] for entry in merges),
            "rapid_flips": sum(entry["rapid_flips"] for entry in merges),
            "rtf": statistics.median([entry["rtf"] for entry in entries]),
            "rss": max(entry["peak_rss_mb"] for entry in entries),
            "wer": [entry.get("wer") for entry in entries if entry.get("wer") is not None],
        }

    cpp = aggregate("whisper-cpp")
    fw = aggregate("faster-whisper")

    def wer_text(values: list[float]) -> str:
        return ", ".join(f"{value:.4f}" for value in values) if values else "n/a (English audio)"

    lines += [
        f"| WER (Turkish recordings) | {wer_text(cpp['wer'])} | {wer_text(fw['wer'])} |",
        f"| timestamp granularity | word-piece tokens, ms offsets | native word intervals |",
        f"| words | {cpp['words']} | {fw['words']} |",
        f"| monotonicity errors | {cpp['monotonicity_errors']} | {fw['monotonicity_errors']} |",
        f"| zero-duration words | {cpp['zero_duration']} | {fw['zero_duration']} |",
        f"| overlapping word timestamps | {cpp['overlaps']} | {fw['overlaps']} |",
        f"| unresolved after merge | {cpp['unresolved']} | {fw['unresolved']} |",
        f"| assigned words | {cpp['assigned']} | {fw['assigned']} |",
        f"| suspicious speaker flips | {cpp['rapid_flips']} | {fw['rapid_flips']} |",
        f"| median RTF | {cpp['rtf']:.4f} | {fw['rtf']:.4f} |",
        f"| peak RSS | {cpp['rss']} MB | {fw['rss']} MB |",
        "",
        "## Per-recording results",
        "",
        "| Recording | Engine | Words | Mono. errors | Zero-dur | Unresolved (known/auto) | Turns | Short turns | Rapid flips | Agreement | RTF |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    def known_mode(merges: dict) -> dict | None:
        for mode, payload in merges.items():
            if mode.startswith("known_"):
                return payload
        return None

    for recording in report["recordings"]:
        for engine, entry in recording["engines"].items():
            known = known_mode(entry["merges"])
            auto = entry["merges"]["auto_0.80"]
            agreement = None
            for payload in entry["merges"].values():
                if payload.get("assignment_agreement"):
                    agreement = payload["assignment_agreement"]["agreement"]
                    break
            lines.append(
                f"| {recording['id']} | {engine} | {entry['words']} | "
                f"{entry['timestamp_health']['monotonicity_errors']} | "
                f"{entry['timestamp_health']['zero_duration_words']} | "
                f"{known['unresolved_words'] if known else 'n/a'} / {auto['unresolved_words']} | "
                f"{len(auto['turns'])} | {auto['short_turns']} | {auto['rapid_flips']} | "
                f"{f'{agreement:.2%}' if agreement is not None else 'n/a'} | {entry['rtf']:.4f} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- Speaker labels stay anonymous (`speaker_<n>`); no identities are inferred.",
        "- Agreement is measured only where reference RTTM turns exist (VoxConverse), "
        "using the best cluster→reference permutation, and is a dataset-ground-truth "
        "diagnostic, not a modelled metric.",
        "- whisper.cpp token timestamps come from the standard heuristic (`t_dtw = -1`); "
        "DTW token timestamps were not computed because the DTW model is not part of the "
        "benchmark assets.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute cached STT outputs")
    parser.add_argument("--turkish-only", action="store_true")
    parser.add_argument("--voxconverse-only", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {
            "diarization": "pyannote 3.0 + TitaNet Small",
            "auto_threshold": TITANET_THRESHOLD,
            "boundary_tolerance_seconds": 0.25,
            "short_turn_seconds": 1.0,
            "threads": THREADS,
        },
        "recordings": [],
    }

    if not args.voxconverse_only:
        log("Turkish controlled recordings (single speaker)")
        for recording in TURKISH_RECORDINGS:
            audio_path = DATA_DIR / recording["file_id"] / "processing.wav"
            log(f"  recording {recording['id']}")
            report["recordings"].append(
                run_recording(
                    recording,
                    audio_path=audio_path,
                    audio_container_path=f"/audio/{recording['file_id']}/processing.wav",
                    mounts=[f"{DATA_DIR}:/audio:ro"],
                    reference_turns=None,
                    reference_speakers=1,
                    force=args.force,
                )
            )

    if not args.turkish_only:
        log("VoxConverse multi-speaker recordings (English, reference RTTM)")
        for recording in VOXCONVERSE_RECORDINGS:
            audio_path = VOXCONVERSE_AUDIO / f"{recording['file_id']}.wav"
            reference_turns = parse_rttm(VOXCONVERSE_RTTM / f"{recording['file_id']}.rttm")
            reference_speakers = len({turn["speaker"] for turn in reference_turns})
            log(f"  recording {recording['id']} ({reference_speakers} reference speakers)")
            report["recordings"].append(
                run_recording(
                    recording,
                    audio_path=audio_path,
                    audio_container_path=f"/audio/{recording['file_id']}.wav",
                    mounts=[f"{VOXCONVERSE_AUDIO}:/audio:ro"],
                    reference_turns=reference_turns,
                    reference_speakers=reference_speakers,
                    force=args.force,
                )
            )

    (RESULTS_DIR / "merge.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "merge.md").write_text(markdown(report), encoding="utf-8")
    log(f"wrote {RESULTS_DIR / 'merge.json'}")
    log(f"wrote {RESULTS_DIR / 'merge.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
