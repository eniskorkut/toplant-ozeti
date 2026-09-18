#!/usr/bin/env python3
"""Host-side dscore scoring for each swept configuration (isolated dscore image)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
REPO_ROOT = HARNESS.parent.parent
DIAR_DIR = REPO_ROOT / "benchmarks" / "diarization"
DATASETS = DIAR_DIR / "datasets" / "voxconverse"
RESULTS = HARNESS / "results"

DSCORE_IMAGE = "mi-bench-dscore:local"
CALIBRATION = ["qpylu", "fxgvy", "szsyz", "rtvuw", "gwtwd", "bwzyf"]
VALIDATION = ["whmpa", "bkwns", "syiwe", "jiqvr", "jyirt", "wjhgf"]


def score(stage: str, key: str, files: list[str], *, ignore_overlaps: bool) -> dict:
    command = [
        "docker", "run", "--rm",
        "-e", "PYTHONPATH=/opt/dscore",
        "-v", f"{DATASETS}:/dataset:ro",
        "-v", f"{HARNESS}:/bench:ro",
        "-v", f"{DIAR_DIR}:/scoring:ro",
        DSCORE_IMAGE,
        "python", "/scoring/score_rttm.py",
        "--reference", *[f"/dataset/voxconverse/dev/{file_id}.rttm" for file_id in files],
        "--system", *[f"/bench/results/rttm/{stage}/{key}/{file_id}.rttm" for file_id in files],
        "--collar", "0.25",
    ]
    if ignore_overlaps:
        command.append("--ignore-overlaps")
    completed = subprocess.run(command, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if completed.returncode != 0:
        raise SystemExit(f"scoring failed for {stage}/{key}:\n{completed.stderr[-1500:]}")
    for line in reversed(completed.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise SystemExit(f"no JSON from scoring for {stage}/{key}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["calibration", "validation"], required=True)
    args = parser.parse_args()

    files = CALIBRATION if args.stage == "calibration" else VALIDATION
    rttm_root = RESULTS / "rttm" / args.stage
    keys = sorted(path.name for path in rttm_root.iterdir() if path.is_dir())
    output: dict = {"stage": args.stage, "files": files, "configs": {}}

    for key in keys:
        primary = score(args.stage, key, files, ignore_overlaps=False)
        diagnostic = score(args.stage, key, files, ignore_overlaps=True)
        output["configs"][key] = {
            "der": primary["global"]["der"],
            "jer": primary["global"]["jer"],
            "der_overlap_ignored": diagnostic["global"]["der"],
        }
        print(f"{key}: DER {primary['global']['der']:.2f}% | JER {primary['global']['jer']:.2f}%", flush=True)

    (RESULTS / f"scoring-{args.stage}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {RESULTS / f'scoring-{args.stage}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
