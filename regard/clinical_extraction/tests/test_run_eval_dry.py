"""Smoke-test eval harness without API calls."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FEATURE = Path(__file__).resolve().parents[1]
RUN_EVAL = FEATURE / "evals" / "scripts" / "run_eval.py"
ROOT = FEATURE.parent.parent


def test_dry_run_with_stress() -> None:
    proc = subprocess.run(
        [sys.executable, str(RUN_EVAL), "--dry-run", "--include-stress"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    metrics = json.loads(proc.stdout)
    assert metrics["schema_pass_rate"] == 1.0
    assert metrics["stress_pass_rate"] == 1.0
