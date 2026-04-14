#!/usr/bin/env python3
"""Periodic failure review helper: cluster top-N error patterns from ``predictions.jsonl``.

Emits Markdown ticket stubs with ``case_id`` samples (no raw model bodies unless
``--include-raw-snippets``). Intended for pilot/production triage (playbook §6.1).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _bucket(row: dict[str, Any]) -> str:
    if not row.get("parse_ok"):
        err = row.get("error") or "parse_failed"
        return re.sub(r"\s+", " ", str(err))[:120]
    if row.get("stress_ok") is False:
        return "stress_check_failed"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=10,
        dest="top_n",
        help="Top N non-ok clusters by count",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write Markdown (default: stdout)",
    )
    parser.add_argument(
        "--include-raw-snippets",
        action="store_true",
        help="Include truncated raw model output in report (may contain sensitive text).",
    )
    args = parser.parse_args()

    buckets: dict[str, list[str]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for row in _load_jsonl(args.predictions_jsonl):
        b = _bucket(row)
        if b == "ok":
            continue
        counts[b] += 1
        cid = str(row.get("case_id", ""))
        if cid and len(buckets[b]) < 5:
            buckets[b].append(cid)

    top = counts.most_common(args.top_n)
    lines = [
        "# Failure review summary",
        "",
        f"Source: `{args.predictions_jsonl}`",
        "",
        "## Top clusters (non-ok)",
        "",
    ]
    for key, cnt in top:
        lines.append(f"### `{cnt}` × — {key}")
        lines.append("")
        lines.append("Sample `case_id`s: " + ", ".join(f"`{x}`" for x in buckets.get(key, [])))
        lines.append("")
        lines.append("- [ ] Ticket: root cause + prompt/workflow version + owner")
        lines.append("")
    if args.include_raw_snippets:
        lines.append("> Raw snippets omitted by default; re-run with flag after access review.")
        lines.append("")

    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
