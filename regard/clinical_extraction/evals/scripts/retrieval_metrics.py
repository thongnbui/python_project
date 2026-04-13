#!/usr/bin/env python3
"""Compute precision@k for RAG cases using gold_chunk_ids on retrieved_chunks order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 1.0
    top = retrieved_ids[:k]
    hits = sum(1 for g in gold_ids if g in top)
    return hits / len(gold_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "gold"
        / "cases_v2026-04-13.jsonl",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    scores: list[float] = []
    for row in _load_jsonl(args.dataset):
        exp = row.get("expected") or {}
        gold_ids = exp.get("gold_chunk_ids")
        if not gold_ids:
            continue
        chunks = row.get("retrieved_chunks") or []
        retrieved_ids = [str(c.get("chunk_id", "")) for c in chunks]
        scores.append(precision_at_k(retrieved_ids, gold_ids, args.k))

    if not scores:
        print("No rows with gold_chunk_ids; nothing to score.", file=sys.stderr)
        return 0

    avg = sum(scores) / len(scores)
    print(json.dumps({"n": len(scores), f"precision_at_{args.k}": round(avg, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
