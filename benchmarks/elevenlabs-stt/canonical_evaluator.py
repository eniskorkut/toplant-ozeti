#!/usr/bin/env python3
"""Canonical Evaluator re-export for benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.canonical_evaluator import (  # noqa: E402
    AggregatedMetrics,
    PolicySpec,
    RecordingMetrics,
    aggregate_macro,
    aggregate_micro,
    build_policies,
    evaluate_timeline,
    load_cache,
    load_reference_turns,
    run_full_calibration,
    simulate_recording,
)

PRIVATE_DIR = SCRIPT_DIR / "results" / "private"

__all__ = [
    "AggregatedMetrics",
    "PolicySpec",
    "RecordingMetrics",
    "aggregate_macro",
    "aggregate_micro",
    "build_policies",
    "evaluate_timeline",
    "load_cache",
    "load_reference_turns",
    "run_full_calibration",
    "simulate_recording",
    "PRIVATE_DIR",
]
