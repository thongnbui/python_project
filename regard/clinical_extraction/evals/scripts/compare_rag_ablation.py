#!/usr/bin/env python3
"""Compare two ``rag_ablation.py`` ``metrics.json`` files (mean precision per top_k).

Exit 1 if any shared ``top_k`` regresses by more than ``--delta`` (absolute).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_k(sweep: list[dict[str, Any]]) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for row in sweep:
        k = int(row["top_k"])
        key = f"mean_precision_at_{k}"
        out[k] = row.get(key)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_metrics_json", type=Path)
    parser.add_argument("candidate_metrics_json", type=Path)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()

    a = _load(args.baseline_metrics_json)
    b = _load(args.candidate_metrics_json)
    ka = _by_k(a.get("sweep") or [])
    kb = _by_k(b.get("sweep") or [])
    keys = sorted(set(ka) & set(kb))
    if not keys:
        print("No overlapping top_k in sweeps", file=sys.stderr)
        return 1
    bad = False
    for k in keys:
        va, vb = ka[k], kb[k]
        if va is None or vb is None:
            continue
        if vb + args.delta < va:
            print(
                f"Regression at top_k={k}: baseline {va} candidate {vb} (delta {args.delta})",
                file=sys.stderr,
            )
            bad = True
    if bad:
        return 1
    print(json.dumps({"ok": True, "compared_top_k": keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
