#!/usr/bin/env python3
"""Compare two metrics.json files (baseline vs new run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.pop("_comment", None)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Older / reference metrics.json")
    parser.add_argument("current", type=Path, help="New metrics.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable diff object",
    )
    args = parser.parse_args()

    for p in (args.baseline, args.current):
        if not p.is_file():
            print(f"Not found: {p}", file=sys.stderr)
            return 1

    a = _read_metrics(args.baseline)
    b = _read_metrics(args.current)
    keys = (
        "n_cases",
        "schema_pass_rate",
        "mean_entity_recall",
        "stress_pass_rate",
    )
    rows: list[dict[str, Any]] = []
    for k in keys:
        av, bv = a.get(k), b.get(k)
        delta = None
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            delta = round(bv - av, 6)
        rows.append(
            {"metric": k, "baseline": av, "current": bv, "delta": delta}
        )

    if args.json:
        print(json.dumps({"rows": rows, "latency_baseline": a.get("latency_ms"), "latency_current": b.get("latency_ms")}, indent=2))
    else:
        print(f"baseline: {args.baseline}")
        print(f"current:  {args.current}")
        for r in rows:
            d = r["delta"]
            ds = f"{d:+.4f}" if d is not None else "n/a"
            print(f"  {r['metric']}: {r['baseline']} -> {r['current']} ({ds})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
