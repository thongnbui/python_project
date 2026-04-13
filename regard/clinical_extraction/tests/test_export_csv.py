"""export_predictions_csv smoke test."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXPORT = ROOT / "regard" / "clinical_extraction" / "evals" / "scripts" / "export_predictions_csv.py"


def test_export_writes_csv(tmp_path: Path) -> None:
    pred = tmp_path / "p.jsonl"
    pred.write_text(
        '{"case_id": "x", "parse_ok": true, "entity_recall": 1.0, "stress_ok": null, '
        '"latency_ms": 1.0, "error": null, "raw": "{}"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    proc = subprocess.run(
        [sys.executable, str(EXPORT), str(pred), "-o", str(out)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "x"


def test_export_with_gold_merge(tmp_path: Path) -> None:
    gold = tmp_path / "g.jsonl"
    gold.write_text(
        '{"case_id": "x", "input": {"chart_excerpt": "Short chart text here."}, '
        '"expected": {"entities": [{"type": "lab", "value": "1"}]}, '
        '"rubric_tags": ["t"], "notes_for_judge": "note"}\n',
        encoding="utf-8",
    )
    pred = tmp_path / "p.jsonl"
    pred.write_text(
        '{"case_id": "x", "parse_ok": true, "entity_recall": 1.0, "stress_ok": null, '
        '"latency_ms": 1.0, "error": null, "raw": "{}"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(EXPORT),
            str(pred),
            "-o",
            str(out),
            "--gold-jsonl",
            str(gold),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["gold_notes_for_judge"] == "note"
    assert "entities" in rows[0]["gold_expected_json"]
    assert "Short chart" in rows[0]["gold_chart_excerpt"]
