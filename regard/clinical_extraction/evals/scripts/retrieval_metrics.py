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


def iter_gold_chunk_rows(path: Path) -> Iterator[tuple[list[str], list[str]]]:
    """Yield ``(retrieved_chunk_ids, gold_chunk_ids)`` for rows that define gold retrieval."""
    for row in _load_jsonl(path):
        exp = row.get("expected") or {}
        gold_ids = exp.get("gold_chunk_ids")
        if not gold_ids:
            continue
        chunks = row.get("retrieved_chunks") or []
        retrieved_ids = [str(c.get("chunk_id", "")) for c in chunks]
        yield retrieved_ids, list(gold_ids)


def mean_precision_for_k(path: Path, k: int) -> tuple[int, float]:
    """Return (n_rows, mean precision@k) over rows with ``gold_chunk_ids``."""
    scores: list[float] = []
    for retrieved_ids, gold_ids in iter_gold_chunk_rows(path):
        scores.append(precision_at_k(retrieved_ids, gold_ids, k))
    if not scores:
        return 0, 0.0
    return len(scores), sum(scores) / len(scores)


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

    n, avg = mean_precision_for_k(args.dataset, args.k)
    if not n:
        print("No rows with gold_chunk_ids; nothing to score.", file=sys.stderr)
        return 0

    print(json.dumps({"n": n, f"precision_at_{args.k}": round(avg, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
