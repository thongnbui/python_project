#!/usr/bin/env python3
"""Replay a fixture to validate workflow packaging (no live LLM).

Emits OpenTelemetry spans per ``workflow.yaml`` (``{prefix}.workflow.run`` and
``{prefix}.step.<tool_name>``) with **hashed** ``trace_id`` only — no chart or query text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from regard.clinical_extraction.agents.tracing import (
    get_otel_prefix,
    get_workflow_tracer,
    hash_identifier,
    trace_safe_attributes,
)


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workflow_caps(feature_root: Path) -> dict[str, Any]:
    wf = feature_root / "agents" / "workflow.yaml"
    return yaml.safe_load(wf.read_text(encoding="utf-8"))


def run_fixture(fixture: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    """Validate fixture against caps; emit OTel spans (safe attributes only)."""
    prefix = get_otel_prefix(caps)
    tracer = get_workflow_tracer()
    raw_tid = str(fixture.get("trace_id") or "unknown")
    tid_h = hash_identifier(raw_tid)
    calls = fixture.get("tool_calls", [])
    max_calls = caps.get("caps", {}).get("max_tool_calls", 8)
    allowed = set(caps.get("tools_allowlist", []))

    with tracer.start_as_current_span(
        f"{prefix}.workflow.run",
        attributes=trace_safe_attributes(
            trace_id_hash=tid_h,
            tool_call_count=len(calls),
        ),
    ):
        if len(calls) > max_calls:
            return {"ok": False, "error": "fixture exceeds max_tool_calls"}
        for i, c in enumerate(calls):
            name = str(c.get("name", "unknown"))
            with tracer.start_as_current_span(
                f"{prefix}.step.{name}",
                attributes=trace_safe_attributes(
                    trace_id_hash=tid_h,
                    step_index=i,
                    tool_name=name,
                ),
            ):
                if name not in allowed:
                    return {
                        "ok": False,
                        "error": f"tool not allowlisted: {name}",
                    }
    return {
        "ok": True,
        "final": fixture.get("final"),
        "trace_id": fixture.get("trace_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).resolve().parent / "fixtures" / "trace_ce-003.json",
    )
    args = parser.parse_args()

    caps = load_workflow_caps(args.feature_root)
    fixture = load_fixture(args.fixture)
    result = run_fixture(fixture, caps)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
