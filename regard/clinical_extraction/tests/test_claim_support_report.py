"""Tests for claim_support_report grounding heuristics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = ROOT / "regard" / "clinical_extraction" / "evals" / "scripts" / "claim_support_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("claim_support_report", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_substring_supported() -> None:
    mod = _load_module()
    corpus = "Patient has Type 2 diabetes mellitus. Home metformin."
    assert mod.claim_supported("Type 2 diabetes mellitus", corpus, token_fallback=True)
    assert mod.claim_supported("metformin", corpus, token_fallback=True)


def test_unsupported_without_fallback_tokens() -> None:
    mod = _load_module()
    corpus = "Patient stable."
    assert not mod.claim_supported("Pluto is a planet", corpus, token_fallback=False)


def test_token_fallback_penicillin_allergy() -> None:
    mod = _load_module()
    corpus = "Allergies: penicillin (rash). Home meds: aspirin."
    assert mod.claim_supported("penicillin allergy", corpus, token_fallback=True)
    assert not mod.claim_supported("penicillin allergy", corpus, token_fallback=False)


def test_cli_on_synthetic_predictions(tmp_path: Path) -> None:
    gold = ROOT / "regard" / "clinical_extraction" / "evals" / "gold" / "cases_v2026-04-13.jsonl"
    raw_ok = json.dumps(
        {
            "entities": [{"type": "problem", "value": "Type 2 diabetes mellitus"}],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        },
        ensure_ascii=False,
    )
    raw_bad = json.dumps(
        {
            "entities": [{"type": "problem", "value": "Made-up syndrome XYZ"}],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        },
        ensure_ascii=False,
    )
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": "ce-001",
                        "parse_ok": True,
                        "raw": raw_ok,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "case_id": "ce-002",
                        "parse_ok": True,
                        "raw": raw_bad,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(pred),
            "--gold-jsonl",
            str(gold),
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
    assert data["aggregate"]["unsupported_claims"] == 1
    assert data["aggregate"]["total_claims"] == 2


def test_cli_threshold_exit(tmp_path: Path) -> None:
    gold = ROOT / "regard" / "clinical_extraction" / "evals" / "gold" / "cases_v2026-04-13.jsonl"
    raw_bad = json.dumps(
        {
            "entities": [{"type": "problem", "value": "Made-up syndrome XYZ"}],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        },
        ensure_ascii=False,
    )
    pred = tmp_path / "predictions.jsonl"
    pred.write_text(
        json.dumps(
            {"case_id": "ce-001", "parse_ok": True, "raw": raw_bad},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            str(pred),
            "--gold-jsonl",
            str(gold),
            "--max-unsupported-rate",
            "0.0",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 1
