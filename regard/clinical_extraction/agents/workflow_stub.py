#!/usr/bin/env python3
"""Replay a fixture to validate workflow packaging (no live LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_workflow_caps(feature_root: Path) -> dict[str, Any]:
    wf = feature_root / "agents" / "workflow.yaml"
    return yaml.safe_load(wf.read_text(encoding="utf-8"))


def run_fixture(fixture: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    calls = fixture.get("tool_calls", [])
    max_calls = caps.get("caps", {}).get("max_tool_calls", 8)
    if len(calls) > max_calls:
        return {"ok": False, "error": "fixture exceeds max_tool_calls"}
    allowed = set(caps.get("tools_allowlist", []))
    for c in calls:
        if c.get("name") not in allowed:
            return {"ok": False, "error": f"tool not allowlisted: {c.get('name')}"}
    return {"ok": True, "final": fixture.get("final"), "trace_id": fixture.get("trace_id")}


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
