"""Tests for llm_judge_eval (dry proxy + calibration math)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "regard" / "clinical_extraction" / "evals" / "scripts" / "llm_judge_eval.py"
_GOLD = ROOT / "regard" / "clinical_extraction" / "evals" / "gold" / "cases_v2026-04-13.jsonl"


def _load_judge_mod():
    spec = importlib.util.spec_from_file_location("llm_judge_eval", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_cohen_kappa_perfect() -> None:
    mod = _load_judge_mod()
    k = mod._cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2])
    assert abs(k - 1.0) < 1e-6


def test_cohen_kappa_no_agreement() -> None:
    mod = _load_judge_mod()
    k = mod._cohen_kappa([0, 0], [2, 2])
    assert k < 1.0


def test_linear_weighted_kappa_perfect() -> None:
    mod = _load_judge_mod()
    k = mod._linear_weighted_kappa([0, 1, 2], [0, 1, 2])
    assert abs(k - 1.0) < 1e-6


def test_linear_weighted_matches_unweighted_when_perfect() -> None:
    mod = _load_judge_mod()
    a = [0, 1, 2, 0]
    b = [0, 1, 2, 0]
    assert abs(mod._cohen_kappa(a, b) - mod._linear_weighted_kappa(a, b)) < 1e-9


def test_dry_run_subprocess(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "entities": [{"type": "problem", "value": "Type 2 diabetes mellitus"}],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        },
        ensure_ascii=False,
    )
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "case_id": "ce-001",
                "parse_ok": True,
                "entity_recall": 1.0,
                "stress_ok": None,
                "raw": raw,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    human = tmp_path / "human.jsonl"
    human.write_text('{"case_id": "ce-001", "score": 2}\n', encoding="utf-8")
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(pred),
            "--gold-jsonl",
            str(_GOLD),
            "--human-scores",
            str(human),
            "--dry-run",
            "-o",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["per_case"][0]["score"] == 2
    assert data["calibration"]["n_overlap"] == 1
    assert data["calibration"]["match_rate"] == 1.0
    assert abs(data["calibration"]["cohens_kappa"] - 1.0) < 1e-6
    assert abs(data["calibration"]["cohens_kappa_linear_weighted"] - 1.0) < 1e-6


def test_dry_run_mismatch_kappa(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "entities": [{"type": "problem", "value": "Type 2 diabetes mellitus"}],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        },
        ensure_ascii=False,
    )
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "case_id": "ce-001",
                "parse_ok": True,
                "entity_recall": 1.0,
                "stress_ok": None,
                "raw": raw,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    human = tmp_path / "human.jsonl"
    human.write_text('{"case_id": "ce-001", "score": 0}\n', encoding="utf-8")
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(pred),
            "--gold-jsonl",
            str(_GOLD),
            "--human-scores",
            str(human),
            "--dry-run",
            "-o",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["per_case"][0]["score"] == 2
    assert data["calibration"]["match_rate"] == 0.0
    assert "cohens_kappa_linear_weighted" in data["calibration"]


def test_min_calibration_n_gate_fails(tmp_path: Path) -> None:
    raw = json.dumps(
        {
            "entities": [{"type": "problem", "value": "Type 2 diabetes mellitus"}],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        },
        ensure_ascii=False,
    )
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {
                "case_id": "ce-001",
                "parse_ok": True,
                "entity_recall": 1.0,
                "stress_ok": None,
                "raw": raw,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    human = tmp_path / "human.jsonl"
    human.write_text('{"case_id": "ce-001", "score": 2}\n', encoding="utf-8")
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(pred),
            "--gold-jsonl",
            str(_GOLD),
            "--human-scores",
            str(human),
            "--min-calibration-n",
            "5",
            "--dry-run",
            "-o",
            str(out),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data.get("calibration_gate_failed") is True
    assert data["calibration"]["n_overlap"] == 1


def test_min_calibration_n_requires_human_scores(tmp_path: Path) -> None:
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        '{"case_id": "ce-001", "parse_ok": true, "raw": "{}"}\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--min-calibration-n",
            "3",
            str(pred),
            "--dry-run",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 2
