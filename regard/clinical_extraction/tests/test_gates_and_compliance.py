"""Prompt stress gate, failure review, RAG ablation compare, SLO enforcement, feature flags."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

FEATURE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


def test_compliance_and_config_files_exist() -> None:
    assert (FEATURE / "compliance" / "phi_minimization.yaml").is_file()
    assert (FEATURE / "compliance" / "retention_policy.yaml").is_file()
    assert (FEATURE / "config" / "feature_flags.yaml").is_file()
    assert (FEATURE / "evals" / "gates" / "prompt_stress_baseline.txt").is_file()
    assert (FEATURE / "docs" / "templates" / "PLAYBOOK_SIGNOFFS.md").is_file()
GATE = FEATURE / "evals" / "scripts" / "verify_prompt_stress_gate.py"
FAIL = FEATURE / "evals" / "scripts" / "failure_review_summarize.py"
CMP = FEATURE / "evals" / "scripts" / "compare_rag_ablation.py"
STUB = FEATURE / "agents" / "workflow_stub.py"
RUN_EVAL = FEATURE / "evals" / "scripts" / "run_eval.py"


def test_verify_prompt_stress_gate_ok() -> None:
    proc = subprocess.run(
        [sys.executable, str(GATE), "--feature-root", str(FEATURE)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr


def test_failure_review_summarize(tmp_path: Path) -> None:
    pred = tmp_path / "p.jsonl"
    pred.write_text(
        '{"case_id": "a", "parse_ok": false, "error": "boom"}\n'
        '{"case_id": "b", "parse_ok": false, "error": "boom"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "r.md"
    proc = subprocess.run(
        [sys.executable, str(FAIL), str(pred), "-o", str(out), "-n", "5"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0
    assert "boom" in out.read_text(encoding="utf-8")


def test_compare_rag_ablation_no_regression(tmp_path: Path) -> None:
    rep = {
        "sweep": [
            {"top_k": 5, "n_cases": 2, "mean_precision_at_5": 0.5},
        ]
    }
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps(rep), encoding="utf-8")
    b.write_text(json.dumps(rep), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(CMP), str(a), str(b), "--delta", "0.01"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0


def test_workflow_stub_enforce_slo() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(STUB),
            "--feature-root",
            str(FEATURE),
            "--fixtures-dir",
            str(FEATURE / "agents" / "fixtures"),
            "--enforce-slo",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr


def test_run_eval_logs_feature_flag() -> None:
    rid = f"ff{uuid.uuid4().hex[:10]}"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_EVAL),
            "--dry-run",
            "--max-cases",
            "1",
            "--force",
            "--run-id",
            rid,
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(ROOT),
            "REGARD_FF_FALLBACK_SINGLE_SHOT": "1",
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "regard_feature_flags" in proc.stderr
    assert "fallback_single_shot" in proc.stderr
