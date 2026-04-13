#!/usr/bin/env python3
"""Export eval predictions.jsonl to CSV for human review (spreadsheet, triage)."""

from __future__ import annotations

import argparse
import csv
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


def _trunc_chart(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    s = str(text).replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"


def _load_gold_by_case_id(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Later files override earlier on duplicate case_id."""
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        if not p.is_file():
            print(f"Warning: gold file not found: {p}", file=sys.stderr)
            continue
        for row in _load_jsonl(p):
            cid = row.get("case_id")
            if cid:
                out[str(cid)] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_jsonl",
        type=Path,
        help="Path to predictions.jsonl from a run_eval output folder",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: same stem as input, .csv)",
    )
    parser.add_argument(
        "--raw-max-chars",
        type=int,
        default=800,
        help="Truncate raw model output column to this length (0 = omit)",
    )
    parser.add_argument(
        "--gold-jsonl",
        action="append",
        default=[],
        metavar="PATH",
        help="Gold dataset JSONL (repeat for multiple: main + stress). Merges by case_id.",
    )
    parser.add_argument(
        "--chart-excerpt-chars",
        type=int,
        default=200,
        metavar="N",
        help="When using --gold-jsonl, include gold_chart_excerpt from input.chart_excerpt "
        "(truncated; 0 = omit column). Default: 200.",
    )
    args = parser.parse_args()

    if not args.predictions_jsonl.is_file():
        print(f"Not found: {args.predictions_jsonl}", file=sys.stderr)
        return 1

    gold_by_id = _load_gold_by_case_id([Path(p) for p in args.gold_jsonl])

    out = args.output or args.predictions_jsonl.with_suffix(".csv")
    gold_fields = []
    if gold_by_id and args.chart_excerpt_chars > 0:
        gold_fields.append("gold_chart_excerpt")
    if gold_by_id:
        gold_fields.extend(
            [
                "gold_expected_json",
                "gold_rubric_tags",
                "gold_notes_for_judge",
            ]
        )
    fieldnames = (
        ["case_id", *gold_fields, "parse_ok", "entity_recall", "stress_ok", "latency_ms", "error", "raw_excerpt"]
        if gold_by_id
        else [
            "case_id",
            "parse_ok",
            "entity_recall",
            "stress_ok",
            "latency_ms",
            "error",
            "raw_excerpt",
        ]
    )

    rows: list[dict[str, object]] = []
    with args.predictions_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw = str(row.get("raw", ""))
            if args.raw_max_chars and len(raw) > args.raw_max_chars:
                raw = raw[: args.raw_max_chars] + "…"
            err = row.get("error") or ""
            if isinstance(err, str) and len(err) > 300:
                err = err[:300] + "…"

            cid = str(row.get("case_id", ""))
            g = gold_by_id.get(cid)
            rec: dict[str, object] = {
                "case_id": cid,
                "parse_ok": row.get("parse_ok"),
                "entity_recall": row.get("entity_recall"),
                "stress_ok": row.get("stress_ok"),
                "latency_ms": row.get("latency_ms"),
                "error": err,
                "raw_excerpt": raw if args.raw_max_chars != 0 else "",
            }
            if gold_by_id:
                inp = (g or {}).get("input") or {}
                chart = inp.get("chart_excerpt", "")
                exp = (g or {}).get("expected")
                tags = (g or {}).get("rubric_tags")
                notes = (g or {}).get("notes_for_judge")
                merged: dict[str, object] = {
                    "case_id": cid,
                    "parse_ok": rec["parse_ok"],
                    "entity_recall": rec["entity_recall"],
                    "stress_ok": rec["stress_ok"],
                    "latency_ms": rec["latency_ms"],
                    "error": rec["error"],
                    "raw_excerpt": rec["raw_excerpt"],
                }
                if args.chart_excerpt_chars > 0:
                    merged["gold_chart_excerpt"] = _trunc_chart(
                        chart, args.chart_excerpt_chars
                    )
                merged["gold_expected_json"] = (
                    json.dumps(exp, ensure_ascii=False) if exp is not None else ""
                )
                merged["gold_rubric_tags"] = (
                    json.dumps(tags, ensure_ascii=False) if tags is not None else ""
                )
                merged["gold_notes_for_judge"] = str(notes or "")
                # preserve column order via fieldnames
                ordered: dict[str, object] = {"case_id": cid}
                for key in fieldnames:
                    if key != "case_id" and key in merged:
                        ordered[key] = merged[key]
                rec = ordered
            rows.append(rec)

    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(str(out.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
