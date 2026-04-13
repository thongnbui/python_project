"""Agent replay metrics: trajectory, duplicates, task match, tool-error recovery (fixture-driven)."""

from __future__ import annotations

import json
import statistics
from typing import Any


def tool_call_signature(call: dict[str, Any]) -> str:
    """Stable string for duplicate detection (name + sorted request JSON)."""
    name = str(call.get("name", ""))
    req = call.get("request")
    if req is None:
        req_s = ""
    else:
        req_s = json.dumps(req, sort_keys=True, ensure_ascii=False)
    return f"{name}\x1f{req_s}"


def duplicate_call_stats(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Count identical tool signatures and longest consecutive duplicate run."""
    if not tool_calls:
        return {
            "unique_signatures": 0,
            "total_calls": 0,
            "duplicate_extra_calls": 0,
            "max_consecutive_duplicate_run": 0,
        }
    sigs = [tool_call_signature(c) for c in tool_calls]
    from collections import Counter

    ctr = Counter(sigs)
    total = len(sigs)
    unique = len(ctr)
    duplicate_extra = total - unique

    max_run = 1
    run = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1

    return {
        "unique_signatures": unique,
        "total_calls": total,
        "duplicate_extra_calls": duplicate_extra,
        "max_consecutive_duplicate_run": max_run,
    }


def count_tool_errors(tool_calls: list[dict[str, Any]]) -> int:
    """Count calls with ``outcome == \"error\"`` (optional field; default ok)."""
    n = 0
    for c in tool_calls:
        if str(c.get("outcome", "ok")).lower() == "error":
            n += 1
    return n


def final_matches_expected(
    final: Any,
    expected: Any,
) -> bool:
    """Exact JSON equality for structured ``final`` vs ``expected_final``."""
    return json.dumps(final, sort_keys=True, ensure_ascii=False) == json.dumps(
        expected, sort_keys=True, ensure_ascii=False
    )


def task_success_metrics(
    *,
    ok: bool,
    final: Any,
    expected_final: Any | None,
) -> dict[str, Any]:
    if expected_final is None:
        return {
            "task_match": "skipped",
            "task_success": None,
        }
    match = ok and final_matches_expected(final, expected_final)
    return {
        "task_match": "exact" if match else "mismatch",
        "task_success": bool(match),
    }


def tool_recovery_metrics(
    *,
    ok: bool,
    tool_calls: list[dict[str, Any]],
    recovery_expected: bool | None,
) -> dict[str, Any]:
    """If fixture declares recovery, check error then eventual success."""
    err_n = count_tool_errors(tool_calls)
    out: dict[str, Any] = {
        "tool_error_calls": err_n,
        "recovery_expected": bool(recovery_expected) if recovery_expected is not None else None,
    }
    if recovery_expected and err_n > 0:
        out["recovery_success"] = bool(ok)
    elif err_n > 0:
        out["recovery_success"] = bool(ok)
    else:
        out["recovery_success"] = None
    return out


def human_effort_from_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Pass-through for optional pilot fields (no PHI in eval harness)."""
    meta = fixture.get("metrics_meta") or {}
    ed = meta.get("human_edit_distance")
    tta = meta.get("time_to_accept_ms")
    return {
        "human_edit_distance": ed,
        "time_to_accept_ms": tta,
    }


def trajectory_percentiles(trajectory_lengths: list[int]) -> dict[str, float | None]:
    """p50 / p95 of trajectory length (step count) across runs."""
    if not trajectory_lengths:
        return {"p50": None, "p95": None, "n": 0}
    s = sorted(trajectory_lengths)
    n = len(s)
    p50 = s[n // 2]
    idx = max(0, int(0.95 * (n - 1)))
    p95 = s[idx]
    return {"p50": float(p50), "p95": float(p95), "n": float(n)}


def aggregate_batch_trajectories(lengths: list[int]) -> dict[str, Any]:
    """Distribution summary for regression checks (p95 flag)."""
    pct = trajectory_percentiles(lengths)
    if not lengths:
        return {"trajectory_tool_calls": pct, "mean": None}
    return {
        "trajectory_tool_calls": pct,
        "mean": round(statistics.mean(lengths), 4),
        "max": max(lengths),
        "min": min(lengths),
    }
