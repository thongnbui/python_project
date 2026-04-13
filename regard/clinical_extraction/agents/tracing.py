"""OpenTelemetry helpers: stable span names, hashed identifiers — no raw PHI in attributes.

Span naming follows ``workflow.yaml`` → ``tracing.otel_span_prefix`` (default
``regard.clinical_extraction``):

* ``{prefix}.workflow.run`` — one root span per fixture / request.
* ``{prefix}.step.<tool_name>`` — one child span per tool call (e.g. ``retrieve_chart``).

Attributes are limited to: ``trace.id_hash`` (SHA-256 hex of ``trace_id`` string),
``step.index``, ``tool.name``, and numeric counts. Do **not** attach chart text,
queries, or patient identifiers.
"""

from __future__ import annotations

import hashlib
from typing import Any

from opentelemetry import trace

DEFAULT_OTEL_PREFIX = "regard.clinical_extraction"


def hash_identifier(value: str, *, nbytes: int = 8) -> str:
    """Return a stable hex digest prefix suitable for span attributes (not reversible)."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[: nbytes * 2]


def get_otel_prefix(caps: dict[str, Any]) -> str:
    """Read ``tracing.otel_span_prefix`` from workflow caps (see ``workflow.yaml``)."""
    tr = caps.get("tracing") or {}
    return str(tr.get("otel_span_prefix") or DEFAULT_OTEL_PREFIX)


def trace_safe_attributes(
    *,
    trace_id_hash: str,
    step_index: int | None = None,
    tool_name: str | None = None,
    tool_call_count: int | None = None,
) -> dict[str, Any]:
    """Build attribute dict that is safe to export (no PHI)."""
    out: dict[str, Any] = {"trace.id_hash": trace_id_hash}
    if step_index is not None:
        out["step.index"] = step_index
    if tool_name is not None:
        out["tool.name"] = tool_name
    if tool_call_count is not None:
        out["tool_call.count"] = tool_call_count
    return out


def get_workflow_tracer() -> trace.Tracer:
    """Tracer for the clinical extraction agent workflow."""
    return trace.get_tracer("regard.clinical_extraction.workflow", "0.1.0")
