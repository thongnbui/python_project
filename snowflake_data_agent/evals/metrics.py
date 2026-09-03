"""LLM-as-judge metrics inspired by the course RAG Triad + efficiency stats.

Uses OpenAI directly (no TruLens dependency) so the harness stays lightweight
and works with the OpenAI tool-calling agent rather than LangGraph/TruGraph.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import openai

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "gpt-4o")


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response (tolerant of fences)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {"score": 0.0, "reason": f"Unparseable judge output: {text[:400]}"}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"score": 0.0, "reason": f"Unparseable judge output: {text[:400]}"}


def _judge(
    client: openai.OpenAI,
    *,
    system: str,
    user: str,
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    payload = _extract_json(response.choices[0].message.content or "")
    score = payload.get("score", 0.0)
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = 0.0
    score_f = max(0.0, min(1.0, score_f))
    return {
        "score": score_f,
        "reason": str(payload.get("reason", "")).strip(),
    }


def session_context_block(
    creds: Optional[dict[str, Any]] = None,
    active: Optional[dict[str, Any]] = None,
) -> str:
    """Known session facts the agent may cite without a tool call.

    Login default database/schema and any active schema set mid-run are part of
    the system prompt, so groundedness judges should treat them as allowed
    evidence (e.g. labeling the login DB as \"your login default\").
    """
    if not creds and not active:
        return ""
    lines: list[str] = ["[session]"]
    if creds:
        if creds.get("database"):
            lines.append(f"login_default_database: {creds['database']}")
        if creds.get("schema"):
            lines.append(f"login_default_schema: {creds['schema']}")
        else:
            lines.append("login_default_schema: (none pinned; explore all schemas)")
        if creds.get("warehouse"):
            lines.append(f"warehouse: {creds['warehouse']}")
        if creds.get("role"):
            lines.append(f"role: {creds['role']}")
    if active and active.get("database") and active.get("schema"):
        lines.append(
            f"active_context: {active['database']}.{active['schema']}"
        )
    return "\n".join(lines)


# Prefer evidence-bearing tools when the judge context budget is tight.
_TOOL_PRIORITY: dict[str, int] = {
    "read_query": 100,
    "describe_table": 90,
    "display_chart": 85,
    "list_databases": 40,
    "list_schemas": 35,
    "list_tables": 45,  # needed for inventory groundedness; compacted below
    "set_active_schema": 20,
    "append_insight": 10,
}

# Soft caps after compaction. Inventory/describe are compacted to name lists
# first so these limits rarely truncate away object names.
_TOOL_BODY_CAPS: dict[str, int] = {
    "list_databases": 2000,
    "list_schemas": 2000,
    "list_tables": 8000,
    "set_active_schema": 200,
    "describe_table": 6000,
    "read_query": 3000,
    "display_chart": 800,
    "append_insight": 400,
}


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"
    return text[: limit - 1] + "…"


def _compact_list_tables(text: str, arguments: Optional[dict[str, Any]]) -> str:
    """Collapse verbose list_tables YAML into a full table-name inventory."""
    names = re.findall(r"(?m)^\s*-?\s*TABLE_NAME:\s*(\S+)\s*$", text or "")
    if not names:
        names = re.findall(r'"TABLE_NAME"\s*:\s*"([^"]+)"', text or "")
    if not names:
        return text or ""
    args = arguments or {}
    db = args.get("database") or ""
    schema = args.get("schema") or ""
    if db and schema:
        labeled = [f"{db}.{schema}.{n}" for n in names]
    else:
        labeled = names
    return f"tables({len(labeled)}): " + ", ".join(labeled)


def _compact_describe_table(text: str, arguments: Optional[dict[str, Any]]) -> str:
    """Collapse describe_table YAML into column name + type pairs."""
    cols = re.findall(
        r"(?m)^\s*-?\s*COLUMN_NAME:\s*(\S+)\s*\n"
        r"(?:\s*COLUMN_DEFAULT:.*\n)?"
        r"\s*IS_NULLABLE:\s*\S+\s*\n"
        r"\s*DATA_TYPE:\s*(\S+)",
        text or "",
    )
    if not cols:
        names = re.findall(r"(?m)^\s*-?\s*COLUMN_NAME:\s*(\S+)\s*$", text or "")
        cols = [(n, "?") for n in names]
    if not cols:
        return text or ""
    args = arguments or {}
    table = args.get("table_name") or ""
    header = (
        f"table={table} columns({len(cols)}): "
        if table
        else f"columns({len(cols)}): "
    )
    body = ", ".join(f"{name}:{dtype}" for name, dtype in cols)
    return header + body


def _format_step_piece(step: dict[str, Any]) -> tuple[str, str, int]:
    """Return ``(tool_name, formatted_piece, priority)`` for one step."""
    name = step.get("tool", "tool")
    priority = _TOOL_PRIORITY.get(name, 50)
    body_cap = _TOOL_BODY_CAPS.get(name, 1200)
    arguments = step.get("arguments") or {}

    if step.get("error"):
        piece = f"[{name} ERROR] {_truncate(str(step['error']), body_cap)}"
    elif "chart" in step:
        chart = step["chart"]
        data = chart.get("data") or []
        sample = json.dumps(data[:8], default=str)
        piece = (
            f"[display_chart] type={chart.get('chart_type')} "
            f"title={chart.get('title')} points={len(data)} "
            f"x={chart.get('x')} y={chart.get('y')} "
            f"sample={_truncate(sample, body_cap)}"
        )
    else:
        body = step.get("text") or ""
        if step.get("rows") is not None:
            body = json.dumps(step["rows"][:50], default=str)
        elif name == "list_tables":
            body = _compact_list_tables(body, arguments)
        elif name == "describe_table":
            body = _compact_describe_table(body, arguments)
        piece = f"[{name}] {_truncate(body, body_cap)}"
    return name, piece, priority


def tool_contexts(steps: list[dict[str, Any]], limit_chars: int = 16000) -> str:
    """Flatten MCP/tool step outputs into a judge context string.

    Packs high-signal tools (``read_query``, ``describe_table``, charts) first.
    ``list_tables`` / ``describe_table`` payloads are compacted to name lists so
    large schemas still fit without starving later SQL evidence.
    """
    if not steps:
        return "(no tool context)"

    formatted: list[tuple[int, int, str]] = []  # (priority, index, piece)
    for idx, step in enumerate(steps):
        _name, piece, priority = _format_step_piece(step)
        formatted.append((priority, idx, piece))

    selected_idxs: set[int] = set()
    used = 0
    for _priority, idx, piece in sorted(
        formatted, key=lambda t: (-t[0], t[1])
    ):
        sep = 2 if selected_idxs else 0
        if used + sep + len(piece) > limit_chars:
            remaining = limit_chars - used - sep
            if remaining >= 80:
                piece = _truncate(piece, remaining)
                selected_idxs.add(idx)
                formatted[idx] = (formatted[idx][0], idx, piece)
                used += sep + len(piece)
            continue
        selected_idxs.add(idx)
        used += sep + len(piece)

    chunks = [piece for _p, idx, piece in formatted if idx in selected_idxs]
    return "\n\n".join(chunks) if chunks else "(no tool context)"


def judge_context(
    steps: list[dict[str, Any]],
    *,
    creds: Optional[dict[str, Any]] = None,
    active: Optional[dict[str, Any]] = None,
    limit_chars: int = 16000,
) -> str:
    """Tool outputs plus allowed session facts for groundedness judges."""
    session = session_context_block(creds, active)
    tool_budget = limit_chars
    if session:
        tool_budget = max(2000, limit_chars - len(session) - 2)
    tools = tool_contexts(steps, limit_chars=tool_budget)
    if session:
        return f"{session}\n\n{tools}"
    return tools


def efficiency_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic process metrics from the tool trace."""
    tools = [s.get("tool") for s in steps if s.get("tool")]
    errors = [s for s in steps if s.get("error")]
    charts = [s for s in steps if "chart" in s]
    return {
        "tool_call_count": len(tools),
        "unique_tools": sorted(set(t for t in tools if t)),
        "error_count": len(errors),
        "chart_count": len(charts),
        "had_errors": bool(errors),
    }


def score_answer_relevance(
    client: openai.OpenAI,
    question: str,
    answer: str,
    context: str = "",
) -> dict[str, Any]:
    """RAG Triad: does the answer address the question?"""
    context_block = ""
    if context.strip():
        context_block = (
            "\n\nSupporting context (session + tools; may confirm which schema "
            f"a bare table name belongs to):\n{context}"
        )
    return _judge(
        client,
        system=(
            "You evaluate answer relevance for a Snowflake data agent. "
            "Score 0-1 how well the answer addresses the user's question. "
            "If the user scopes a database.schema and asks to pick a table or "
            "column inside it, choosing any real table/column from that schema "
            "is correct — a bare TABLE or TABLE.COLUMN name is fine when "
            "context shows it lives in the scoped schema. Do NOT penalize for "
            "omitting the three-part prefix unless the question required a "
            "specific named table. "
            "Return JSON: {\"score\": <float>, \"reason\": \"...\"}."
        ),
        user=f"Question:\n{question}\n\nAnswer:\n{answer}{context_block}",
    )


def score_groundedness(
    client: openai.OpenAI,
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """RAG Triad: is the answer supported by tool + session context?"""
    return _judge(
        client,
        system=(
            "You evaluate groundedness for a Snowflake data agent. Score 0-1 "
            "whether claims in the answer are supported by the provided context. "
            "Context may include a [session] block (login default database/schema, "
            "active schema) AND tool/SQL outputs. Session facts ARE allowed "
            "evidence — e.g. calling the login_default_database \"your login "
            "default\" is grounded. Compacted list_tables lines like "
            "\"tables(N): db.schema.T1, db.schema.T2, ...\" and describe_table "
            "lines like \"columns(N): COL:TYPE, ...\" are complete inventories — "
            "treat every listed name as present in context. Penalize invented "
            "tables, columns, numbers, or databases not present in session or "
            "tool context. "
            "Return JSON: {\"score\": <float>, \"reason\": \"...\"}."
        ),
        user=(
            f"Question:\n{question}\n\nAllowed context (session + tools):\n"
            f"{context}\n\nAnswer:\n{answer}"
        ),
    )


def score_context_relevance(
    client: openai.OpenAI,
    question: str,
    context: str,
) -> dict[str, Any]:
    """RAG Triad: was the retrieved/tool context relevant to the question?"""
    return _judge(
        client,
        system=(
            "You evaluate context relevance. Score 0-1 whether the tool results "
            "are useful for answering the question (even if incomplete). "
            "Return JSON: {\"score\": <float>, \"reason\": \"...\"}."
        ),
        user=f"Question:\n{question}\n\nTool context:\n{context}",
    )


def score_execution_efficiency(
    client: openai.OpenAI,
    question: str,
    steps: list[dict[str, Any]],
    efficiency: dict[str, Any],
) -> dict[str, Any]:
    """GPA-inspired: was the tool sequence reasonably efficient?"""
    summary = []
    for step in steps:
        name = step.get("tool", "?")
        args = step.get("arguments") or {}
        compact = {k: args[k] for k in list(args)[:4]}
        summary.append(f"- {name}({json.dumps(compact, default=str)[:200]})")
    trace = "\n".join(summary) or "(no tools)"
    return _judge(
        client,
        system=(
            "You evaluate execution efficiency of a Snowflake tool-using agent. "
            "Score 0-1: high if tools are necessary and non-redundant; low if "
            "there are repeated failures, pointless calls, or clear waste. "
            f"Observed stats: {json.dumps(efficiency)}. "
            "Return JSON: {\"score\": <float>, \"reason\": \"...\"}."
        ),
        user=f"Question:\n{question}\n\nTool trace:\n{trace}",
    )


def check_expectations(
    steps: list[dict[str, Any]],
    expect: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Hard checks from golden_queries.json ``expect`` blocks."""
    expect = expect or {}
    efficiency = efficiency_metrics(steps)
    tools_used = set(efficiency["unique_tools"])
    preferred = set(expect.get("preferred_tools") or [])
    min_calls = int(expect.get("min_tool_calls") or 0)
    require_chart = bool(expect.get("require_chart"))

    checks = {
        "min_tool_calls": efficiency["tool_call_count"] >= min_calls,
        "used_any_preferred_tool": (
            not preferred or bool(tools_used & preferred)
        ),
        "require_chart": (not require_chart) or efficiency["chart_count"] > 0,
        "no_tool_errors": not efficiency["had_errors"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "efficiency": efficiency,
    }


def score_trace(
    client: openai.OpenAI,
    *,
    question: str,
    answer: str,
    steps: list[dict[str, Any]],
    expect: Optional[dict[str, Any]] = None,
    judge: bool = True,
    creds: Optional[dict[str, Any]] = None,
    active: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run expectation checks + optional RAG Triad / efficiency judges."""
    expectations = check_expectations(steps, expect)
    tool_only = tool_contexts(steps)
    # Groundedness may use session facts; context-relevance stays tool-focused.
    grounded_context = judge_context(steps, creds=creds, active=active)
    result: dict[str, Any] = {
        "expectations": expectations,
        "context_preview": grounded_context[:1500],
    }
    if not judge:
        return result

    result["answer_relevance"] = score_answer_relevance(
        client, question, answer, grounded_context
    )
    result["groundedness"] = score_groundedness(
        client, question, answer, grounded_context
    )
    result["context_relevance"] = score_context_relevance(
        client, question, tool_only
    )
    result["execution_efficiency"] = score_execution_efficiency(
        client, question, steps, expectations["efficiency"]
    )
    triad = [
        result["answer_relevance"]["score"],
        result["groundedness"]["score"],
        result["context_relevance"]["score"],
    ]
    result["rag_triad_mean"] = sum(triad) / len(triad)
    return result
