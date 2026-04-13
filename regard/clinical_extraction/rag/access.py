"""Apply ``access_model.yaml`` role filters to retrieved chunks (offline helper)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FEATURE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCESS_PATH = FEATURE_ROOT / "rag" / "access_model.yaml"


def load_access_model(path: Path | None = None) -> dict[str, Any]:
    p = path or DEFAULT_ACCESS_PATH
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _chunk_doc_type(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    return str(meta.get("doc_type") or chunk.get("doc_type") or "").strip()


def filter_chunks_by_role(
    chunks: list[dict[str, Any]],
    role: str,
    *,
    access_model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return chunks whose ``doc_type`` is allowed for ``role``."""
    model = access_model or load_access_model()
    roles = model.get("roles") or {}
    spec = roles.get(role) or roles.get(model.get("default_role", "clinician"))
    if not spec:
        return list(chunks)
    allowed = {str(x).strip() for x in (spec.get("allowed_doc_types") or [])}
    if not allowed:
        return list(chunks)
    out: list[dict[str, Any]] = []
    for ch in chunks:
        dt = _chunk_doc_type(ch)
        if dt in allowed:
            out.append(ch)
    return out
