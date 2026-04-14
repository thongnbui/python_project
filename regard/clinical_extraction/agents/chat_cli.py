#!/usr/bin/env python3
"""Minimal multi-turn **CLI agent** (tool-calling) for the progress-note workflow.

**Live** (``OPENAI_API_KEY``): OpenAI Chat Completions with ``retrieve_chart`` and
``draft_progress_note``. Tool **implementations are mocked** (no real vector DB)—chunks
and draft text are synthetic so you can exercise the loop safely. On startup, env vars
are loaded from the repo-root ``.env`` (if present), then ``clinical_extraction/.env``
with override—same order as ``run_eval.py``—so you can keep the key only in
``regard/clinical_extraction/.env``.

**Offline** (``--offline``): no API; each user line triggers a canned retrieve + draft
simulation (good for CI and smoke tests).

Do not paste real PHI into this terminal; use synthetic chart language only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

FEATURE_ROOT = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Use retrieve_chart when you need "
    "evidence from the chart, then draft_progress_note when asked to write a note. "
    "Keep answers concise. Tools return mock data (not a real EHR)."
)

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_chart",
            "description": "Retrieve relevant chart chunks for a clinical query.",
            "parameters": {
                "type": "object",
                "required": ["query", "top_k"],
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    "time_from": {"type": "string"},
                    "time_to": {"type": "string"},
                    "doc_types": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_progress_note",
            "description": "Draft a progress note section from cited chunk ids.",
            "parameters": {
                "type": "object",
                "required": ["sections", "context_chunk_ids"],
                "properties": {
                    "sections": {"type": "array", "items": {"type": "string"}},
                    "context_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
]


def _load_caps() -> dict[str, Any]:
    p = FEATURE_ROOT / "agents" / "workflow.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def mock_retrieve_chart(args: dict[str, Any]) -> dict[str, Any]:
    q = str(args.get("query", ""))[:80]
    return {
        "chunks": [
            {
                "chunk_id": "cli-mock-1",
                "text": f"(Mock chunk) Related excerpt for query: {q!r}",
                "metadata": {"doc_type": "progress_note"},
            },
            {
                "chunk_id": "cli-mock-2",
                "text": "(Mock chunk) Home meds reviewed; continue monitoring.",
                "metadata": {"doc_type": "progress_note"},
            },
        ]
    }


def mock_draft_progress_note(args: dict[str, Any]) -> dict[str, Any]:
    ids = args.get("context_chunk_ids") or []
    return {
        "markdown": "## Assessment (mock)\n- Summarized from chunks: "
        + ", ".join(str(i) for i in ids)
        + "\n",
        "citations": [{"chunk_id": str(i)} for i in ids],
    }


def dispatch_tool(name: str, raw_args: str) -> dict[str, Any]:
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        args = {}
    if name == "retrieve_chart":
        return mock_retrieve_chart(args)
    if name == "draft_progress_note":
        return mock_draft_progress_note(args)
    return {"error": f"unknown tool {name}"}


def offline_turn(user_text: str) -> None:
    """Single-turn canned pipeline without OpenAI."""
    print("\n[offline] assistant (simulated):")
    r = mock_retrieve_chart({"query": user_text, "top_k": 5})
    print("  retrieve_chart ->", json.dumps(r, indent=2)[:600])
    d = mock_draft_progress_note(
        {"sections": ["Assessment"], "context_chunk_ids": ["cli-mock-1"]}
    )
    print("  draft_progress_note ->", json.dumps(d, indent=2)[:600])


def _load_openai_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
    load_dotenv(FEATURE_ROOT / ".env", override=True)


def _run_tool_rounds(
    client: Any,
    model: str,
    messages: list[dict[str, Any]],
    max_tool_rounds: int,
) -> None:
    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tcs = msg.tool_calls or []
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
        }
        if tcs:
            assistant["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tcs
            ]
        messages.append(assistant)

        if not tcs:
            if msg.content:
                print(f"\nassistant> {msg.content}")
            break

        for tc in tcs:
            name = tc.function.name
            out = dispatch_tool(name, tc.function.arguments)
            print(f"\n[tool {name}] {json.dumps(out, ensure_ascii=False)[:400]}…")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(out, ensure_ascii=False),
                }
            )


def run_live_loop(
    *,
    model: str,
    max_tool_rounds: int,
    lines: list[str],
) -> None:
    from openai import OpenAI

    _load_openai_env()
    client = OpenAI()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for user_text in lines:
        user_text = user_text.strip()
        if not user_text:
            continue
        print(f"\nuser> {user_text}")
        messages.append({"role": "user", "content": user_text})
        _run_tool_rounds(client, model, messages, max_tool_rounds)


def _read_script(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--script",
        type=Path,
        default=None,
        help="Non-interactive: run each non-empty line as one user turn, then exit.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="No OpenAI; print mock tool outputs per line (--script or stdin).",
    )
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=None,
        help="Max assistant+tool cycles per user message (default from workflow.yaml caps).",
    )
    args = parser.parse_args()
    # Load `.env` before any OPENAI_API_KEY check (feature `.env` overrides repo root).
    _load_openai_env()

    caps = _load_caps().get("caps") or {}
    default_rounds = min(int(caps.get("max_tool_calls", 8)), 12)
    max_tool_rounds = args.max_tool_rounds or default_rounds

    if args.script is not None:
        lines = _read_script(args.script)
    elif not sys.stdin.isatty():
        lines = [ln.strip() for ln in sys.stdin.read().splitlines()]
    else:
        # Interactive TTY session (conversation state kept across lines in live mode).
        print(
            "Clinical extraction agent CLI. /quit or EOF to exit. "
            "Use --offline for mock tools without OPENAI_API_KEY.",
            file=sys.stderr,
        )
        if not args.offline and not os.environ.get("OPENAI_API_KEY", "").strip():
            print(
                "Missing OPENAI_API_KEY. Use --offline, export the key, or add it to "
                "clinical_extraction/.env (see .env.example).",
                file=sys.stderr,
            )
            return 2
        if not args.offline:
            from openai import OpenAI

            client = OpenAI()
            messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        while True:
            try:
                ln = input("you> ").strip()
            except EOFError:
                break
            if ln.lower() in ("/quit", "exit", ":q"):
                break
            if not ln:
                continue
            if args.offline:
                print(f"\nuser> {ln}")
                offline_turn(ln)
                continue
            print(f"\nuser> {ln}")
            messages.append({"role": "user", "content": ln})
            _run_tool_rounds(client, args.model, messages, max_tool_rounds)
        return 0

    filtered = [ln for ln in lines if ln.strip()]
    if not filtered:
        print("No non-empty input lines.", file=sys.stderr)
        parser.print_help()
        return 1

    if args.offline:
        for ln in filtered:
            print(f"\nuser> {ln}")
            offline_turn(ln)
        return 0

    if not os.environ.get("OPENAI_API_KEY", "").strip():
        print(
            "Missing OPENAI_API_KEY (or use --offline). "
            "Copy clinical_extraction/.env.example → .env and set the key.",
            file=sys.stderr,
        )
        return 2

    run_live_loop(
        model=args.model,
        max_tool_rounds=max_tool_rounds,
        lines=filtered,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
