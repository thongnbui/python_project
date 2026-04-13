#!/usr/bin/env python3
"""Grounding report: label each extracted entity value vs chart + retrieved chunks + quotes.

Implements a **deterministic** hallucination / claim-support protocol for eval triage:
each entity is *supported* if it can be matched against the merged source text (normalized
substring, with optional token-level fallback). This is not clinical truth—only overlap with
provided context. Use blinded LLM review for production-grade adjudication.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from regard.clinical_extraction.schemas.models import ExtractionOutput

FEATURE_ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_gold_by_case_id(paths: list[Path]) -> dict[str, dict[str, Any]]:
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


def _normalize_entity_text(value: str) -> str:
    """Match run_eval.py: lowercase, collapse whitespace, trim trailing punctuation."""
    s = str(value).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".,;:")


def _build_support_corpus(case: dict[str, Any], pred: ExtractionOutput) -> str:
    parts: list[str] = []
    inp = case.get("input") or {}
    parts.append(str(inp.get("chart_excerpt", "") or ""))
    for ch in case.get("retrieved_chunks") or []:
        if isinstance(ch, dict):
            parts.append(str(ch.get("text", "") or ""))
    for c in pred.citations:
        if isinstance(c, dict):
            parts.append(str(c.get("quote", "") or ""))
    return " ".join(parts)


def _significant_tokens(norm_claim: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9%/.-]+", norm_claim)
        if len(t) >= 2 and t not in {"mg", "ml", "bid", "qhs", "prn"}
    ]


def _token_grounded(token: str, corpus_norm: str) -> bool:
    """Whether *token* appears in *corpus_norm* (substring or common EN plural flip)."""
    if token in corpus_norm:
        return True
    if len(token) < 4:
        return False
    # e.g. allergy ↔ allergies (token not substring of "allergies" without this)
    if token.endswith("y"):
        plural = token[:-1] + "ies"
        if plural in corpus_norm:
            return True
    if token.endswith("ies") and len(token) > 3:
        singular = token[:-3] + "y"
        if singular in corpus_norm:
            return True
    return False


def claim_supported(
    claim: str,
    corpus: str,
    *,
    token_fallback: bool,
) -> bool:
    """Return True if *claim* is plausibly grounded in *corpus* (heuristic)."""
    cn = _normalize_entity_text(claim)
    sn = _normalize_entity_text(corpus)
    if not cn:
        return True
    if cn in sn:
        return True
    if not token_fallback:
        return False
    toks = _significant_tokens(cn)
    if not toks:
        return False
    return all(_token_grounded(t, sn) for t in toks)


def _parse_predictions_path(arg: Path) -> Path:
    if arg.is_dir():
        p = arg / "predictions.jsonl"
        if not p.is_file():
            raise SystemExit(f"No predictions.jsonl under {arg}")
        return p
    if not arg.is_file():
        raise SystemExit(f"Not a file or run directory: {arg}")
    return arg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_jsonl_or_run_dir",
        type=Path,
        help="predictions.jsonl path, or a run directory containing it",
    )
    parser.add_argument(
        "--gold-jsonl",
        action="append",
        default=[],
        metavar="PATH",
        help="Gold JSONL (repeat). Required to supply chart_excerpt + retrieved_chunks.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON report (default: stdout)",
    )
    parser.add_argument(
        "--no-token-fallback",
        action="store_true",
        help="Require normalized substring only (stricter; more false 'unsupported').",
    )
    parser.add_argument(
        "--max-unsupported-rate",
        type=float,
        default=None,
        metavar="R",
        help="If set, exit 1 when unsupported_rate > R (0–1). Ignored if no claims.",
    )
    args = parser.parse_args()
    pred_path = _parse_predictions_path(args.predictions_jsonl_or_run_dir)

    if not args.gold_jsonl:
        args.gold_jsonl = [
            FEATURE_ROOT / "evals" / "gold" / "cases_v2026-04-13.jsonl",
            FEATURE_ROOT / "evals" / "gold" / "stress.jsonl",
        ]
    gold_by_id = _load_gold_by_case_id([Path(p) for p in args.gold_jsonl])

    token_fallback = not args.no_token_fallback
    per_case: list[dict[str, Any]] = []
    total_claims = 0
    unsupported_claims = 0

    for row in _load_jsonl(pred_path):
        case_id = str(row.get("case_id", ""))
        case = gold_by_id.get(case_id)
        entry: dict[str, Any] = {
            "case_id": case_id,
            "parse_ok": row.get("parse_ok"),
            "claims": [],
        }
        if not row.get("parse_ok"):
            entry["skip_reason"] = "parse_failed"
            per_case.append(entry)
            continue
        if case is None:
            entry["skip_reason"] = "missing_gold_case"
            per_case.append(entry)
            continue

        try:
            body = json.loads(row.get("raw") or "{}")
            # run_eval validates after _merge_meta; predictions.jsonl stores pre-merge API text.
            body.setdefault(
                "_meta",
                {"prompt_version": "unknown", "model": "unknown"},
            )
            pred = ExtractionOutput.model_validate(body)
        except Exception as exc:  # noqa: BLE001
            entry["skip_reason"] = "raw_parse_error"
            entry["error"] = str(exc)
            per_case.append(entry)
            continue

        corpus = _build_support_corpus(case, pred)
        for ent in pred.entities:
            val = str(ent.get("value", "") or "")
            et = str(ent.get("type", "") or "")
            ok = claim_supported(val, corpus, token_fallback=token_fallback)
            total_claims += 1
            if not ok:
                unsupported_claims += 1
            entry["claims"].append(
                {
                    "type": et,
                    "value": val,
                    "status": "supported" if ok else "unsupported",
                }
            )
        per_case.append(entry)

    unsupported_rate = (
        unsupported_claims / total_claims if total_claims else None
    )
    report = {
        "predictions": str(pred_path),
        "gold_files": [str(p) for p in args.gold_jsonl],
        "token_fallback": token_fallback,
        "per_case": per_case,
        "aggregate": {
            "total_claims": total_claims,
            "unsupported_claims": unsupported_claims,
            "unsupported_rate": round(unsupported_rate, 6)
            if unsupported_rate is not None
            else None,
        },
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)

    if args.max_unsupported_rate is not None and total_claims:
        if unsupported_rate is not None and unsupported_rate > args.max_unsupported_rate:
            print(
                f"Threshold exceeded: unsupported_rate {unsupported_rate:.4f} "
                f"> {args.max_unsupported_rate}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
