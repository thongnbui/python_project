"""OpenTelemetry span names and PHI safety for workflow_stub."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from regard.clinical_extraction.agents.workflow_stub import (
    load_fixture,
    load_workflow_caps,
    run_fixture,
)

FEATURE = Path(__file__).resolve().parents[1]
FIXTURE = FEATURE / "agents" / "fixtures" / "trace_ce-003.json"

# TracerProvider can only be set once per process; share one in-memory exporter.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = TracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))
trace.set_tracer_provider(_PROVIDER)


@pytest.fixture(autouse=True)
def _clear_spans() -> None:
    _EXPORTER.clear()
    yield


def test_run_fixture_emits_expected_span_names() -> None:
    caps = load_workflow_caps(FEATURE)
    fixture = load_fixture(FIXTURE)
    assert run_fixture(fixture, caps).get("ok") is True
    names = [s.name for s in _EXPORTER.get_finished_spans()]
    assert "regard.clinical_extraction.workflow.run" in names
    assert "regard.clinical_extraction.step.retrieve_chart" in names


def test_span_attributes_contain_hash_not_plaintext_phi() -> None:
    caps = load_workflow_caps(FEATURE)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert run_fixture(fixture, caps).get("ok") is True
    sensitive = "CKD stage 3"
    dumped = json.dumps(
        [
            {
                "name": s.name,
                "attrs": dict(s.attributes or {}),
            }
            for s in _EXPORTER.get_finished_spans()
        ],
        default=str,
    )
    assert sensitive not in dumped
    assert "trace.id_hash" in dumped


def test_disallowlisted_tool_still_emits_spans() -> None:
    caps = load_workflow_caps(FEATURE)
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    fixture["tool_calls"] = [{"name": "forbidden_tool", "request": {}, "response": {}}]
    res = run_fixture(fixture, caps)
    assert res.get("ok") is False
    assert _EXPORTER.get_finished_spans()
