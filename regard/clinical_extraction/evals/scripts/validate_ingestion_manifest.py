#!/usr/bin/env python3
"""Validate an ingestion manifest JSONL: unique (source_version, doc_id) per line.

Each object must include string fields ``source_version`` and ``doc_id`` (idempotency key
from ``rag/ingestion_contract.yaml``).
"""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_jsonl", type=Path)
    args = parser.parse_args()

    seen: set[tuple[str, str]] = set()
    line_no = 0
    for row in _load_jsonl(args.manifest_jsonl):
        line_no += 1
        sv = row.get("source_version")
        did = row.get("doc_id")
        if sv is None or did is None:
            print(f"Line {line_no}: missing source_version or doc_id", file=sys.stderr)
            return 1
        key = (str(sv), str(did))
        if key in seen:
            print(f"Line {line_no}: duplicate idempotency key {key!r}", file=sys.stderr)
            return 1
        seen.add(key)

    print(json.dumps({"ok": True, "unique_documents": len(seen)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
