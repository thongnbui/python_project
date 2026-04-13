#!/usr/bin/env python3
"""Export a single replay fixture as an internal **debug bundle** (JSON on disk).

Intended for on-call / engineering only: includes full tool request/response bodies from the
fixture (may contain sensitive text). Restrict access in production (ACL, encryption, retention).

With ``--redact``, long string values under known keys (``text``, ``query``, ``markdown``) are
replaced by ``sha256:`` hex prefixes for safer handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REDACT_KEYS = frozenset({"text", "query", "markdown", "chart_excerpt"})


def _hash_hint(s: str, n: int = 16) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def redact_fixture(obj: Any, depth: int = 0) -> Any:
    """Recursively redact string fields for known PHI-heavy keys."""
    if depth > 24:
        return "<max_depth>"
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k in _REDACT_KEYS and isinstance(v, str) and len(v) > 8:
                out[k] = _hash_hint(v)
            else:
                out[k] = redact_fixture(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [redact_fixture(x, depth + 1) for x in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="Fixture JSON path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output bundle path",
    )
    parser.add_argument(
        "--redact",
        action="store_true",
        help="Hash long strings for text/query/markdown/chart_excerpt fields",
    )
    args = parser.parse_args()
    raw = json.loads(args.fixture.read_text(encoding="utf-8"))
    body = redact_fixture(raw) if args.redact else raw
    bundle = {
        "bundle_version": 1,
        "source": "regard.clinical_extraction.agents.export_debug_bundle",
        "redacted": bool(args.redact),
        "fixture": body,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
