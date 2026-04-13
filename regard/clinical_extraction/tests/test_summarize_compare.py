"""summarize_run.py and compare_metrics.py smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARIZE = ROOT / "regard" / "clinical_extraction" / "evals" / "scripts" / "summarize_run.py"
COMPARE = ROOT / "regard" / "clinical_extraction" / "evals" / "scripts" / "compare_metrics.py"


def test_summarize_json(tmp_path: Path) -> None:
    pred = tmp_path / "p.jsonl"
    pred.write_text(
        '{"case_id": "a", "parse_ok": true, "entity_recall": 1.0, "stress_ok": null}\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(SUMMARIZE), str(pred), "--format", "json"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["n_cases"] == 1
    assert out["parse_ok_count"] == 1


def test_compare_metrics(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(
        json.dumps({"schema_pass_rate": 1.0, "mean_entity_recall": 0.5}),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps({"schema_pass_rate": 1.0, "mean_entity_recall": 0.8}),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(COMPARE), str(a), str(b)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "0.3" in proc.stdout or "+0.3000" in proc.stdout
