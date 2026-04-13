"""Fail if dry-run metrics regress vs committed baseline_metrics.json."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

FEATURE = Path(__file__).resolve().parents[1]
RUN_EVAL = FEATURE / "evals" / "scripts" / "run_eval.py"
BASELINE = FEATURE / "evals" / "baseline_metrics.json"
ROOT = FEATURE.parent.parent


def test_dry_run_meets_committed_baseline() -> None:
    assert BASELINE.is_file(), f"Missing {BASELINE}"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_EVAL),
            "--dry-run",
            "--include-stress",
            "--baseline-metrics",
            str(BASELINE),
            "--run-id",
            "pytest-baseline-regression",
            "--force",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
