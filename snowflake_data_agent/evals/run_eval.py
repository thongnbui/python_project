#!/usr/bin/env python3
"""CLI for the Snowflake data-agent eval harness.

Examples (from snowflake_data_agent/)::

    # Smoke subset (no LLM judges — cheaper)
    .venv/bin/python -m evals.run_eval --tags smoke --no-judge

    # Full suite with RAG Triad + efficiency judges
    .venv/bin/python -m evals.run_eval --app-version "v1: base"

    # One query
    .venv/bin/python -m evals.run_eval --ids list_databases --no-judge
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python -m evals.run_eval`` from the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.runner import DEFAULT_GOLDEN, DEFAULT_RESULTS_DIR, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run golden-query evals against the Snowflake MCP agent."
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="Path to golden_queries.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for JSON reports",
    )
    parser.add_argument("--model", default=None, help="Agent model override")
    parser.add_argument(
        "--app-version",
        default="v1: base",
        help="Label stored on the report (compare versions over time)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Only run these query ids",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Only run queries that have any of these tags",
    )
    parser.add_argument(
        "--schema",
        default="",
        help="Override schema used for {{schema}} placeholders",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run at most N queries after filtering",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judges; still record traces + expectation checks",
    )
    args = parser.parse_args()

    run_suite(
        golden_path=args.golden,
        results_dir=args.results_dir,
        model=args.model,
        query_ids=args.ids,
        tags=args.tags,
        judge=not args.no_judge,
        app_version=args.app_version,
        schema_override=args.schema,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
