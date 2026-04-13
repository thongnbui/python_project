#!/usr/bin/env python3
"""Print a short summary of predictions.jsonl (and optional metrics.json)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_jsonl",
        type=Path,
        help="predictions.jsonl from evals/runs/<id>/",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional metrics.json from the same run (echoes aggregates)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
    )
    parser.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit with code 1 if any parse_ok false or stress_ok false.",
    )
    args = parser.parse_args()

    if not args.predictions_jsonl.is_file():
        print(f"Not found: {args.predictions_jsonl}", file=sys.stderr)
        return 1

    pred_rows = _load_jsonl(args.predictions_jsonl)
    n = len(pred_rows)
    parse_ok = sum(1 for r in pred_rows if r.get("parse_ok"))
    failed = [str(r.get("case_id", "")) for r in pred_rows if not r.get("parse_ok")]
    recalls = [r["entity_recall"] for r in pred_rows if r.get("entity_recall") is not None]
    stress_flags = [
        r["stress_ok"]
        for r in pred_rows
        if r.get("stress_ok") is not None and str(r.get("case_id", "")).startswith("stress")
    ]
    stress_fail = [
        str(r.get("case_id", ""))
        for r in pred_rows
        if str(r.get("case_id", "")).startswith("stress")
        and r.get("stress_ok") is False
    ]

    summary: dict[str, Any] = {
        "predictions_file": str(args.predictions_jsonl.resolve()),
        "n_cases": n,
        "parse_ok_count": parse_ok,
        "parse_fail_count": n - parse_ok,
        "parse_failed_case_ids": failed,
        "mean_entity_recall": round(mean(recalls), 4) if recalls else None,
        "stress_cases_evaluated": len(stress_flags),
        "stress_pass_count": sum(1 for x in stress_flags if x),
        "stress_failed_case_ids": stress_fail,
    }

    if args.metrics_json and args.metrics_json.is_file():
        mj = json.loads(args.metrics_json.read_text(encoding="utf-8"))
        summary["metrics_json"] = {
            k: mj.get(k)
            for k in (
                "run_id",
                "schema_pass_rate",
                "mean_entity_recall",
                "stress_pass_rate",
                "latency_ms",
                "tokens",
            )
            if k in mj
        }

    if args.format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(f"File: {summary['predictions_file']}")
        print(f"Cases: {n} | parse_ok: {parse_ok} | parse_fail: {n - parse_ok}")
        if failed:
            print(f"  Failed case_ids: {', '.join(failed)}")
        if summary["mean_entity_recall"] is not None:
            print(f"Mean entity_recall (where scored): {summary['mean_entity_recall']}")
        if stress_flags:
            print(
                f"Stress: {summary['stress_pass_count']}/{len(stress_flags)} passed",
                end="",
            )
            if stress_fail:
                print(f" | failed: {', '.join(stress_fail)}")
            else:
                print()
        if "metrics_json" in summary:
            print("metrics.json:", json.dumps(summary["metrics_json"], indent=2))

    if args.strict_exit and (failed or stress_fail):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
