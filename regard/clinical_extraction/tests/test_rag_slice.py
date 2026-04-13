"""RAG playbook slice: access filter, ablation, faithfulness, ingestion manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from regard.clinical_extraction.evals.scripts.retrieval_metrics import mean_precision_for_k
from regard.clinical_extraction.rag.access import filter_chunks_by_role, load_access_model

FEATURE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
GOLD = FEATURE / "evals" / "gold" / "cases_v2026-04-13.jsonl"
TRACE = FEATURE / "agents" / "fixtures" / "trace_ce-003.json"
AB_SCRIPT = FEATURE / "evals" / "scripts" / "rag_ablation.py"
FAITH_SCRIPT = FEATURE / "evals" / "scripts" / "answer_faithfulness.py"
VAL_SCRIPT = FEATURE / "evals" / "scripts" / "validate_ingestion_manifest.py"


def test_access_model_filters_by_role() -> None:
    chunks = [
        {"chunk_id": "1", "text": "a", "metadata": {"doc_type": "problem_list"}},
        {"chunk_id": "2", "text": "b", "metadata": {"doc_type": "lab"}},
    ]
    lim = filter_chunks_by_role(chunks, "researcher_limited")
    assert [c["chunk_id"] for c in lim] == ["2"]
    full = filter_chunks_by_role(chunks, "clinician")
    assert len(full) == 2


def test_load_access_model() -> None:
    m = load_access_model()
    assert "clinician" in (m.get("roles") or {})


def test_mean_precision_for_k_smoke() -> None:
    n, avg = mean_precision_for_k(GOLD, 5)
    assert n >= 1
    assert 0.0 <= avg <= 1.0


def test_rag_ablation_cli() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(AB_SCRIPT),
            "--dataset",
            str(GOLD),
            "--top-k-values",
            "1,5",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert len(data["sweep"]) == 2


def test_answer_faithfulness_cli() -> None:
    proc = subprocess.run(
        [sys.executable, str(FAITH_SCRIPT), str(TRACE)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["sentence_count"] >= 1


def test_ingestion_manifest_ok(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_text(
        '{"source_version":"v1","doc_id":"a"}\n{"source_version":"v1","doc_id":"b"}\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(VAL_SCRIPT), str(p)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0


def test_ingestion_manifest_duplicate(tmp_path: Path) -> None:
    p = tmp_path / "m.jsonl"
    p.write_text(
        '{"source_version":"v1","doc_id":"a"}\n{"source_version":"v1","doc_id":"a"}\n',
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(VAL_SCRIPT), str(p)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 1


def test_rag_ablation_writes_output_dir(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(AB_SCRIPT),
            "--dataset",
            str(GOLD),
            "--top-k-values",
            "3",
            "--output-dir",
            str(tmp_path / "abl"),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0
    assert (tmp_path / "abl" / "metrics.json").is_file()
    assert (tmp_path / "abl" / "plots" / "README.txt").is_file()
