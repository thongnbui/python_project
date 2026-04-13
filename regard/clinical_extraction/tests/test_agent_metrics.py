"""Agent fixture metrics, structured logs, debug bundle (§4.3 / §4.4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from regard.clinical_extraction.agents.export_debug_bundle import redact_fixture
from regard.clinical_extraction.agents.metrics import duplicate_call_stats, final_matches_expected
from regard.clinical_extraction.agents.structured_logging import reset_structured_log_sink
from regard.clinical_extraction.agents.workflow_stub import (
    batch_fixtures_metrics,
    load_fixture,
    load_workflow_caps,
    run_fixture,
)

FEATURE = Path(__file__).resolve().parents[1]
_CAPS = load_workflow_caps(FEATURE)


def test_duplicate_call_stats() -> None:
    calls = [
        {"name": "retrieve_chart", "request": {"q": 1}},
        {"name": "retrieve_chart", "request": {"q": 1}},
        {"name": "draft_progress_note", "request": {}},
    ]
    s = duplicate_call_stats(calls)
    assert s["duplicate_extra_calls"] == 1
    assert s["max_consecutive_duplicate_run"] == 2


def test_expected_final_match() -> None:
    f = load_fixture(FEATURE / "agents" / "fixtures" / "trace_ce-003.json")
    assert final_matches_expected(f["final"], f["expected_final"])


def test_recovery_fixture_metrics() -> None:
    f = load_fixture(FEATURE / "agents" / "fixtures" / "trace_recovery.json")
    r = run_fixture(f, _CAPS)
    assert r["ok"] is True
    assert r["metrics"]["tool_error_calls"] == 1
    assert r["metrics"].get("recovery_success") is True


def test_batch_trajectory_percentiles() -> None:
    paths = sorted((FEATURE / "agents" / "fixtures").glob("*.json"))
    rep = batch_fixtures_metrics(paths, _CAPS)
    agg = rep["aggregate"]["trajectory_tool_calls"]
    assert agg["n"] == len(paths)
    assert agg["p95"] is not None


def test_structured_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reset_structured_log_sink()
    log_path = tmp_path / "steps.ndjson"
    monkeypatch.setenv("REGARD_AGENT_STRUCTURED_LOG", "1")
    monkeypatch.setenv("REGARD_AGENT_LOG_PATH", str(log_path))
    f = load_fixture(FEATURE / "agents" / "fixtures" / "trace_ce-003.json")
    run_fixture(f, _CAPS)
    text = log_path.read_text(encoding="utf-8")
    assert "trace_id_hash" in text
    line = json.loads(text.strip().split("\n")[0])
    assert line["tool_name"] == "retrieve_chart"
    assert "CKD" not in text
    reset_structured_log_sink()


def test_debug_bundle_redact() -> None:
    obj = {"request": {"query": "secret patient note"}, "text": "long " * 10}
    red = redact_fixture(obj)
    assert "secret" not in json.dumps(red)
    assert red["request"]["query"].startswith("sha256:")


def test_max_duplicate_extra_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    script = root / "regard" / "clinical_extraction" / "agents" / "workflow_stub.py"
    fx = FEATURE / "agents" / "fixtures" / "trace_duplicates.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--feature-root",
            str(FEATURE),
            "--fixture",
            str(fx),
            "--max-duplicate-extra",
            "0",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(root)},
    )
    assert proc.returncode == 1
