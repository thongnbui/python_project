"""CLI flags: --case-ids and recall behavior (offline)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_EVAL = ROOT / "regard" / "clinical_extraction" / "evals" / "scripts" / "run_eval.py"


def _run_eval_json(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, str(RUN_EVAL), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    return json.loads(proc.stdout)


def test_case_ids_filters_order() -> None:
    m = _run_eval_json(
        [
            "--dry-run",
            "--include-stress",
            "--case-ids",
            "stress-001,ce-001",
            "--run-id",
            "pytest-case-ids",
            "--force",
        ]
    )
    assert m["n_cases"] == 2


def test_normalize_recall_default_dry_baseline_still_passes() -> None:
    """Normalized recall must not break committed dry baseline."""
    baseline = (
        ROOT
        / "regard"
        / "clinical_extraction"
        / "evals"
        / "baseline_metrics.json"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_EVAL),
            "--dry-run",
            "--include-stress",
            "--baseline-metrics",
            str(baseline),
            "--run-id",
            "pytest-norm-recall",
            "--force",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
