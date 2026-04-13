#!/usr/bin/env python3
"""Replay a fixture to validate workflow packaging (no live LLM).

Emits OpenTelemetry spans per ``workflow.yaml`` (``{prefix}.workflow.run`` and
``{prefix}.step.<tool_name>``) with **hashed** ``trace_id`` only — no chart or query text.

Optional **structured logs** (NDJSON): set ``REGARD_AGENT_STRUCTURED_LOG=1`` and optionally
``REGARD_AGENT_LOG_PATH=...``. Each line includes ``trace_id_hash`` on every step (trace
correlation without raw PHI in shared sinks).

Metrics (§4.3) and observability (§4.4) are returned under ``metrics``; see ``agents/metrics.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from regard.clinical_extraction.agents.metrics import (
    aggregate_batch_trajectories,
    duplicate_call_stats,
    human_effort_from_fixture,
    task_success_metrics,
    tool_recovery_metrics,
)
from regard.clinical_extraction.agents.structured_logging import (
    build_step_record,
    emit_step_log,
)
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
    """Validate fixture against caps; emit OTel spans and optional structured step logs."""
    prefix = get_otel_prefix(caps)
    tracer = get_workflow_tracer()
    raw_tid = str(fixture.get("trace_id") or "unknown")
    tid_h = hash_identifier(raw_tid)
    calls = fixture.get("tool_calls", [])
    max_calls = caps.get("caps", {}).get("max_tool_calls", 8)
    allowed = set(caps.get("tools_allowlist", []))
    expected_final = fixture.get("expected_final")
    recovery_expected = (fixture.get("metrics_expectation") or {}).get("recovery_expected")

    dup = duplicate_call_stats(calls)
    step_logs: list[dict[str, Any]] = []

    structural_ok = True
    err: str | None = None
    with tracer.start_as_current_span(
        f"{prefix}.workflow.run",
        attributes=trace_safe_attributes(
            trace_id_hash=tid_h,
            tool_call_count=len(calls),
        ),
    ):
        if len(calls) > max_calls:
            structural_ok = False
            err = "fixture exceeds max_tool_calls"
        else:
            for i, c in enumerate(calls):
                name = str(c.get("name", "unknown"))
                t0 = time.perf_counter()
                outcome = str(c.get("outcome", "ok")).lower()
                if outcome not in ("ok", "error"):
                    outcome = "ok"
                retry_count = int(c.get("retry_count", 0) or 0)

                with tracer.start_as_current_span(
                    f"{prefix}.step.{name}",
                    attributes=trace_safe_attributes(
                        trace_id_hash=tid_h,
                        step_index=i,
                        tool_name=name,
                    ),
                ):
                    duration_ms = (time.perf_counter() - t0) * 1000.0
                    rec = build_step_record(
                        trace_id_hash=tid_h,
                        step_index=i,
                        tool_name=name,
                        duration_ms=duration_ms,
                        outcome=outcome,
                        retry_count=retry_count,
                        correlation_trace_id_hash=tid_h,
                    )
                    emit_step_log(rec)
                    step_logs.append(rec)

                    if name not in allowed:
                        structural_ok = False
                        err = f"tool not allowlisted: {name}"
                        break

    final = fixture.get("final")
    tm = task_success_metrics(
        ok=structural_ok,
        final=final,
        expected_final=expected_final,
    )
    rec_m = tool_recovery_metrics(
        ok=structural_ok,
        tool_calls=calls,
        recovery_expected=recovery_expected,
    )
    human_m = human_effort_from_fixture(fixture)

    task_ok = True
    if expected_final is not None:
        task_ok = bool(tm.get("task_success"))

    ok = structural_ok and (task_ok if expected_final is not None else True)

    metrics: dict[str, Any] = {
        "trajectory_steps": len(calls),
        "trajectory_tool_calls": len(calls),
        **dup,
        **tm,
        **rec_m,
        "human_effort": human_m,
    }

    return {
        "ok": ok,
        "structural_ok": structural_ok,
        "final": final,
        "trace_id": fixture.get("trace_id"),
        "metrics": metrics,
        "step_logs": step_logs,
        "error": err,
    }


def batch_fixtures_metrics(
    fixture_paths: list[Path],
    caps: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate trajectory lengths for p50/p95 (§4.3 regression helper)."""
    lengths: list[int] = []
    runs: list[dict[str, Any]] = []
    for p in fixture_paths:
        fixture = load_fixture(p)
        r = run_fixture(fixture, caps)
        lengths.append(r["metrics"]["trajectory_steps"])
        runs.append({"fixture": str(p), "ok": r["ok"], "metrics": r["metrics"]})
    agg = aggregate_batch_trajectories(lengths)
    return {"runs": runs, "aggregate": agg}


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
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=None,
        help="Run all *.json fixtures and print aggregate trajectory p50/p95.",
    )
    parser.add_argument(
        "--max-duplicate-extra",
        type=int,
        default=None,
        metavar="N",
        help="Exit 1 if duplicate_extra_calls > N (CI gate).",
    )
    args = parser.parse_args()

    caps = load_workflow_caps(args.feature_root)
    if args.fixtures_dir is not None:
        paths = sorted(args.fixtures_dir.glob("*.json"))
        report = batch_fixtures_metrics(paths, caps)
        print(json.dumps(report, indent=2))
        return 0

    fixture = load_fixture(args.fixture)
    result = run_fixture(fixture, caps)
    print(json.dumps(result, indent=2))
    if args.max_duplicate_extra is not None:
        dup = result.get("metrics", {}).get("duplicate_extra_calls", 0)
        if dup > args.max_duplicate_extra:
            print(
                f"duplicate_extra_calls {dup} > {args.max_duplicate_extra}",
                file=sys.stderr,
            )
            return 1
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
