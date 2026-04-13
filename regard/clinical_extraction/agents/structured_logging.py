"""Structured per-step logs (no PHI in default fields)."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, TextIO

_sink_cache: tuple[str | None, TextIO | None] = (None, None)


def reset_structured_log_sink() -> None:
    """Close cached file handle (tests / subprocess isolation)."""
    global _sink_cache
    old = _sink_cache[1]
    if old is not None and old is not sys.stderr:
        try:
            old.close()
        except OSError:
            pass
    _sink_cache = (None, None)


def _get_sink() -> TextIO | None:
    """Return stderr or a single append-only file handle (cached by path key)."""
    global _sink_cache
    mode = os.environ.get("REGARD_AGENT_STRUCTURED_LOG", "").strip().lower()
    if not mode or mode in ("0", "false", "no"):
        return None
    path = os.environ.get("REGARD_AGENT_LOG_PATH", "").strip()
    key = f"path:{path}" if path else "stderr"
    if _sink_cache[0] == key and _sink_cache[1] is not None:
        return _sink_cache[1]
    if path:
        fp: TextIO = open(path, "a", encoding="utf-8")
    else:
        fp = sys.stderr
    _sink_cache = (key, fp)
    return fp


def emit_step_log(record: dict[str, Any]) -> None:
    """Append one JSON line (NDJSON) when ``REGARD_AGENT_STRUCTURED_LOG`` is enabled."""
    record.setdefault("ts_unix", time.time())
    line = json.dumps(record, ensure_ascii=False) + "\n"
    fp = _get_sink()
    if fp is None:
        return
    try:
        fp.write(line)
        fp.flush()
    except OSError:
        pass


def build_step_record(
    *,
    trace_id_hash: str,
    step_index: int,
    tool_name: str,
    duration_ms: float,
    outcome: str,
    retry_count: int = 0,
    correlation_trace_id_hash: str | None = None,
) -> dict[str, Any]:
    """Fields suitable for observability backends; no chart/query text."""
    rec: dict[str, Any] = {
        "trace_id_hash": trace_id_hash,
        "step_index": step_index,
        "step_name": tool_name,
        "tool_name": tool_name,
        "duration_ms": round(duration_ms, 3),
        "outcome": outcome,
        "retry_count": retry_count,
    }
    if correlation_trace_id_hash:
        rec["correlation"] = {"trace_id_hash": correlation_trace_id_hash}
    return rec
