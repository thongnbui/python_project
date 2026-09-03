"""Headless runner: execute golden queries through ``agent_core.run_agent``."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai
from dotenv import load_dotenv

from agent_core import (
    DEFAULT_MODEL,
    build_server_params,
    build_system_prompt,
    load_mcp_creds_from_env,
    run_agent,
)
from evals.metrics import score_trace
from mcp_client import SnowflakeMCPClient

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_queries.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_creds_from_env() -> dict[str, str]:
    """Build MCP credentials from environment variables."""
    load_dotenv(ROOT / ".env")
    try:
        return load_mcp_creds_from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def load_golden(path: Path = DEFAULT_GOLDEN) -> dict[str, Any]:
    return json.loads(path.read_text())


def render_prompt(template: str, creds: dict[str, str], schema_override: str = "") -> str:
    database = creds["database"]
    schema = schema_override or creds.get("schema") or "PUBLIC"
    return (
        template.replace("{{database}}", database).replace("{{schema}}", schema)
    )


def _make_apply_active_schema(messages: list[dict], creds: dict[str, str]):
    active: dict[str, Optional[str]] = {"database": None, "schema": None}

    def apply_active_schema(database: str, schema: str) -> str:
        active["database"] = database
        active["schema"] = schema
        messages[0]["content"] = build_system_prompt(
            creds, {"database": database, "schema": schema}
        )
        return f"Active context set to {database}.{schema}."

    return apply_active_schema, active


def run_one_query(
    *,
    oai: openai.OpenAI,
    mcp: SnowflakeMCPClient,
    creds: dict[str, str],
    query: dict[str, Any],
    model: str,
    judge: bool = True,
    schema_override: str = "",
) -> dict[str, Any]:
    """Run a single golden query and score the trace."""
    if query.get("requires_schema") and not (
        schema_override or creds.get("schema")
    ):
        return {
            "id": query["id"],
            "skipped": True,
            "reason": "requires_schema but SNOWFLAKE_SCHEMA / --schema not set",
        }

    prompt = render_prompt(query["prompt"], creds, schema_override)
    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt(creds, None)},
        {"role": "user", "content": prompt},
    ]
    apply_active_schema, active = _make_apply_active_schema(messages, creds)
    steps: list[dict] = []

    started = time.time()
    answer = run_agent(
        client=oai,
        mcp_client=mcp,
        model=model,
        messages=messages,
        on_step=steps.append,
        apply_active_schema=apply_active_schema,
    )
    elapsed = time.time() - started

    active_schema = None
    if active.get("database") and active.get("schema"):
        active_schema = {
            "database": active["database"],
            "schema": active["schema"],
        }

    scores = score_trace(
        oai,
        question=prompt,
        answer=answer,
        steps=steps,
        expect=query.get("expect"),
        judge=judge,
        creds=creds,
        active=active_schema,
    )
    return {
        "id": query["id"],
        "skipped": False,
        "tags": query.get("tags") or [],
        "prompt": prompt,
        "answer": answer,
        "steps": steps,
        "elapsed_sec": round(elapsed, 2),
        "model": model,
        "scores": scores,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    ran = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    triad = [
        r["scores"]["rag_triad_mean"]
        for r in ran
        if r.get("scores", {}).get("rag_triad_mean") is not None
    ]
    expect_pass = [
        r["scores"]["expectations"]["passed"]
        for r in ran
        if r.get("scores", {}).get("expectations")
    ]
    return {
        "n_total": len(results),
        "n_ran": len(ran),
        "n_skipped": len(skipped),
        "expect_pass_rate": (
            sum(1 for x in expect_pass if x) / len(expect_pass) if expect_pass else None
        ),
        "rag_triad_mean": (sum(triad) / len(triad) if triad else None),
        "mean_elapsed_sec": (
            sum(r["elapsed_sec"] for r in ran) / len(ran) if ran else None
        ),
    }


def run_suite(
    *,
    golden_path: Path = DEFAULT_GOLDEN,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    model: Optional[str] = None,
    query_ids: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    judge: bool = True,
    app_version: str = "v1: base",
    schema_override: str = "",
    limit: Optional[int] = None,
) -> Path:
    """Run the golden suite and write a JSON report under ``evals/results/``."""
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run evals.")

    creds = load_creds_from_env()
    golden = load_golden(golden_path)
    queries = deepcopy(golden.get("queries") or [])
    if query_ids:
        wanted = set(query_ids)
        queries = [q for q in queries if q["id"] in wanted]
    if tags:
        tagset = set(tags)
        queries = [q for q in queries if tagset & set(q.get("tags") or [])]
    if limit is not None:
        queries = queries[:limit]

    model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    oai = openai.OpenAI()
    mcp = SnowflakeMCPClient(build_server_params(creds))

    results: list[dict[str, Any]] = []
    try:
        for query in queries:
            print(f"→ {query['id']} ...", flush=True)
            result = run_one_query(
                oai=oai,
                mcp=mcp,
                creds=creds,
                query=query,
                model=model,
                judge=judge,
                schema_override=schema_override,
            )
            if result.get("skipped"):
                print(f"  skipped: {result.get('reason')}", flush=True)
            else:
                scores = result["scores"]
                triad = scores.get("rag_triad_mean")
                passed = scores.get("expectations", {}).get("passed")
                print(
                    f"  tools={scores['expectations']['efficiency']['tool_call_count']} "
                    f"expect_pass={passed} rag_triad={triad} "
                    f"({result['elapsed_sec']}s)",
                    flush=True,
                )
            results.append(result)
    finally:
        mcp.close()

    report = {
        "app_version": app_version,
        "model": model,
        "judge": judge,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "database": creds["database"],
        "schema": schema_override or creds.get("schema") or None,
        "summary": summarize(results),
        "results": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = results_dir / f"eval_{stamp}.json"
    # Drop bulky step row payloads in the saved report's optional compact form?
    # Keep full steps for debugging; results/ is gitignored.
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    print(json.dumps(report["summary"], indent=2))
    return out_path
