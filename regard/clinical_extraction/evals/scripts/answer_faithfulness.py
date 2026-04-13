#!/usr/bin/env python3
"""Heuristic answer faithfulness: each sentence vs merged retrieval corpus (no LLM).

Splits ``markdown`` on sentence boundaries and uses the same substring / token grounding
heuristics as ``claim_support_report.py``. Intended for triage and CI on **fixtures**, not
clinical adjudication.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


def _load_claim_support():
    path = Path(__file__).resolve().parent / "claim_support_report.py"
    spec = importlib.util.spec_from_file_location("_csr", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _split_sentences(markdown: str) -> list[str]:
    text = markdown.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _corpus_from_agent_fixture(fixture: dict[str, Any]) -> str:
    parts: list[str] = []
    for call in fixture.get("tool_calls") or []:
        resp = call.get("response") or {}
        for ch in resp.get("chunks") or []:
            if isinstance(ch, dict):
                parts.append(str(ch.get("text", "") or ""))
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture_json",
        type=Path,
        help="Agent replay fixture with tool_calls[].response.chunks and final.markdown",
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
        help="Stricter substring-only grounding (see claim_support_report).",
    )
    args = parser.parse_args()

    fixture = json.loads(args.fixture_json.read_text(encoding="utf-8"))
    final = fixture.get("final") or {}
    md = str(final.get("markdown", "") or "")
    corpus = _corpus_from_agent_fixture(fixture)
    csr = _load_claim_support()
    token_fallback = not args.no_token_fallback

    sentences = _split_sentences(md)
    per_sent: list[dict[str, Any]] = []
    unsupported = 0
    for s in sentences:
        ok = csr.claim_supported(s, corpus, token_fallback=token_fallback)
        if not ok:
            unsupported += 1
        per_sent.append({"sentence": s, "status": "supported" if ok else "unsupported"})

    n = len(sentences)
    rate = unsupported / n if n else None
    report = {
        "fixture": str(args.fixture_json),
        "sentence_count": n,
        "unsupported_sentences": unsupported,
        "unsupported_rate": round(rate, 6) if rate is not None else None,
        "per_sentence": per_sent,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
