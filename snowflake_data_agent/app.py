"""Snowflake Data Explorer — a chat agent backed by an MCP server.

This Streamlit app connects to a Snowflake instance through the
``mcp_snowflake_server`` MCP server and lets you explore databases, schemas,
tables and their data in plain English. An OpenAI model drives the conversation
and calls the MCP tools (``list_databases``, ``list_schemas``, ``list_tables``,
``describe_table``, ``read_query``, ...) to answer your questions.

Run it with::

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mcp import StdioServerParameters

from mcp_client import SnowflakeMCPClient

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_AGENT_STEPS = 8

# Token/size guardrails to stay under the OpenAI tokens-per-minute (TPM) limit.
# Tool results are appended to the conversation and re-sent on every agent step,
# so we cap each result and trim old turns to keep requests small.
# Roughly 4 characters per token.
MAX_TOOL_RESULT_CHARS = 6000  # ~1.5k tokens per tool result
MAX_CONTEXT_CHARS = 48000  # ~12k tokens of history per request (well under 30k)

# The MCP server requires *some* default schema to connect. When the user does
# not pin one, fall back to INFORMATION_SCHEMA (present in every database) so the
# agent is free to explore all schemas rather than being scoped to one.
DEFAULT_SCHEMA = "INFORMATION_SCHEMA"

SYSTEM_PROMPT = """\
You are a Snowflake data exploration assistant. You help the user browse and \
understand the data in their Snowflake account.

You have tools to:
- list databases, schemas, and tables,
- describe a table's columns and types,
- run read-only SELECT queries.

Guidelines:
- To explore broadly, start by listing databases, then schemas, then tables.
- The connection has a default database/schema, but it is only a starting \
context. When the user wants to "explore everything", enumerate ALL schemas in \
the database with list_schemas and inspect tables across them — do not limit \
yourself to the default schema.
- Before querying an unfamiliar table, describe it to learn its columns.
- ALWAYS add a LIMIT (default 100) to exploratory SELECT queries so you never \
pull huge result sets.
- Only issue read-only SELECT statements. Never attempt INSERT/UPDATE/DELETE/DDL.
- Qualify object names as DATABASE.SCHEMA.TABLE when the context is ambiguous.
- When you show rows, summarize what they mean. Keep prose concise.
- If a tool returns an error, explain it and suggest how to fix it (e.g. wrong \
database/schema, missing privileges, warehouse not running).
"""


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def build_server_params() -> tuple[Optional[StdioServerParameters], list[str]]:
    """Build MCP server launch parameters from environment variables.

    Returns a tuple of (params, missing_required_vars).
    """
    account = os.getenv("SNOWFLAKE_ACCOUNT")
    user = os.getenv("SNOWFLAKE_USER")
    password = os.getenv("SNOWFLAKE_PASSWORD")
    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    role = os.getenv("SNOWFLAKE_ROLE")
    database = os.getenv("SNOWFLAKE_DATABASE")
    # SNOWFLAKE_SCHEMA is optional: when omitted we default to INFORMATION_SCHEMA
    # so the agent can explore every schema in the database freely.
    schema = os.getenv("SNOWFLAKE_SCHEMA") or DEFAULT_SCHEMA

    # The mcp_snowflake_server requires a default database to connect (it asserts
    # on it), in addition to the core credentials.
    required = {
        "SNOWFLAKE_ACCOUNT": account,
        "SNOWFLAKE_USER": user,
        "SNOWFLAKE_PASSWORD": password,
        "SNOWFLAKE_WAREHOUSE": warehouse,
        "SNOWFLAKE_DATABASE": database,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return None, missing

    # The server's console script name. ``uvx`` will fetch it on first run.
    command = os.getenv("SNOWFLAKE_MCP_COMMAND", "uvx")
    package = os.getenv("SNOWFLAKE_MCP_PACKAGE", "mcp_snowflake_server")

    args: list[str] = []
    # When launched with uvx, pin dependencies to work around a known
    # pyOpenSSL/cryptography incompatibility (missing X509_V_FLAG_NOTIFY_POLICY).
    # These ``--with`` flags must precede the package name.
    if os.path.basename(command) == "uvx":
        pins = os.getenv("SNOWFLAKE_MCP_UV_WITH", "cryptography<44,setuptools<75")
        for spec in (s.strip() for s in pins.split(",")):
            if spec:
                args += ["--with", spec]

    args += [package]
    args += ["--account", account, "--user", user, "--password", password]
    args += ["--warehouse", warehouse]
    if role:
        args += ["--role", role]
    if database:
        args += ["--database", database]
    if schema:
        args += ["--schema", schema]

    # Pass through the parent environment so ``uvx``/PATH resolve correctly.
    env = {k: v for k, v in os.environ.items()}
    return StdioServerParameters(command=command, args=args, env=env), []


@st.cache_resource(show_spinner="Connecting to Snowflake via MCP...")
def get_mcp_client(cache_key: str) -> SnowflakeMCPClient:
    """Create (and cache) the MCP client for the current connection settings.

    ``cache_key`` is derived from the connection settings so changing them
    invalidates the cached client.
    """
    params, missing = build_server_params()
    if params is None:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )
    return SnowflakeMCPClient(params)


def connection_cache_key() -> str:
    """A stable key that changes when connection-relevant settings change."""
    parts = [
        os.getenv("SNOWFLAKE_ACCOUNT", ""),
        os.getenv("SNOWFLAKE_USER", ""),
        os.getenv("SNOWFLAKE_WAREHOUSE", ""),
        os.getenv("SNOWFLAKE_ROLE", ""),
        os.getenv("SNOWFLAKE_DATABASE", ""),
        os.getenv("SNOWFLAKE_SCHEMA", ""),
        os.getenv("SNOWFLAKE_MCP_PACKAGE", "mcp_snowflake_server"),
    ]
    return "|".join(parts)


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #
def _message_size(message: dict) -> int:
    """Rough character footprint of a message (content + tool call args)."""
    size = len(str(message.get("content") or ""))
    for call in message.get("tool_calls") or []:
        size += len(str(call.get("function", {}).get("arguments", "")))
    return size


def trim_messages(messages: list[dict], budget: int = MAX_CONTEXT_CHARS) -> list[dict]:
    """Keep the system prompt + the most recent messages within a char budget.

    Cuts only at ``user`` message boundaries so we never orphan an assistant
    ``tool_calls`` message from its ``tool`` responses (OpenAI rejects that).
    """
    if not messages:
        return messages
    system = messages[:1] if messages[0]["role"] == "system" else []
    body = messages[len(system):]

    # Indices where a new user turn begins — these are safe places to cut.
    user_starts = [i for i, m in enumerate(body) if m["role"] == "user"]

    total = sum(_message_size(m) for m in system) + sum(_message_size(m) for m in body)
    cut = 0
    for start in user_starts:
        if total <= budget:
            break
        # Drop everything before this user turn.
        dropped = sum(_message_size(m) for m in body[cut:start])
        total -= dropped
        cut = start

    return system + body[cut:]


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Truncate long tool output, keeping a marker so the model knows."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    return (
        f"{head}\n\n... [truncated {len(text) - limit} characters; "
        "narrow the query with column selection or a smaller LIMIT for full data]"
    )


def run_agent(
    client: openai.OpenAI,
    mcp_client: SnowflakeMCPClient,
    model: str,
    messages: list[dict],
    on_step,
) -> str:
    """Run the tool-calling loop until the model produces a final answer.

    ``messages`` is the full OpenAI message history (mutated in place).
    ``on_step`` is a callback ``(tool_name, arguments, result)`` used to render
    intermediate tool calls in the UI.
    Returns the final assistant text.
    """
    tools = mcp_client.openai_tools()

    for _ in range(MAX_AGENT_STEPS):
        response = client.chat.completions.create(
            model=model,
            messages=trim_messages(messages),
            tools=tools,
            temperature=0,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            content = message.content or ""
            messages.append({"role": "assistant", "content": content})
            return content

        # Record the assistant turn that requested tool calls.
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
        )

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            try:
                result = mcp_client.call_tool(name, arguments)
                result_text = result.text or "(no output)"
                on_step(name, arguments, result)
            except Exception as exc:  # noqa: BLE001 - report failures to the model
                result_text = f"Tool call failed: {exc}"
                on_step(name, arguments, None, error=str(exc))

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _truncate(result_text),
                }
            )

    messages.append(
        {
            "role": "assistant",
            "content": "I reached the maximum number of tool calls. "
            "Please refine your question and try again.",
        }
    )
    return messages[-1]["content"]


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def render_step(step: dict) -> None:
    """Render a single tool-call step inside an expander."""
    name = step["tool"]
    label = f"🛠️ `{name}`"
    if step.get("error"):
        label = f"❌ `{name}` failed"
    with st.expander(label, expanded=False):
        if step.get("arguments"):
            st.caption("Arguments")
            st.code(json.dumps(step["arguments"], indent=2), language="json")
        if step.get("error"):
            st.error(step["error"])
            return
        rows = step.get("rows")
        if rows:
            st.caption(f"Result ({len(rows)} rows)")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("Result")
            st.code(step.get("text", ""), language="json")


def step_recorder(store: list[dict]):
    """Build an ``on_step`` callback that records and live-renders steps."""

    def _on_step(name: str, arguments: dict, result, error: Optional[str] = None):
        step: dict[str, Any] = {"tool": name, "arguments": arguments}
        if error is not None:
            step["error"] = error
        elif result is not None:
            step["text"] = result.text
            step["rows"] = result.rows
        store.append(step)
        render_step(step)

    return _on_step


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Snowflake Data Explorer", page_icon="❄️", layout="wide")
st.title("❄️ Snowflake Data Explorer")
st.caption("Chat with your Snowflake data through an MCP server.")

if "openai_messages" not in st.session_state:
    st.session_state.openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "chat" not in st.session_state:
    st.session_state.chat = []  # Display items: {role, content, steps}

# --- Sidebar: connection + tools ----------------------------------------- #
with st.sidebar:
    st.header("Connection")
    _, missing = build_server_params()
    if missing:
        st.error(
            "Missing required settings:\n\n"
            + "\n".join(f"- `{name}`" for name in missing)
            + "\n\nAdd them to a `.env` file (see `.env.example`)."
        )
        st.stop()

    st.write(f"**Account:** `{os.getenv('SNOWFLAKE_ACCOUNT')}`")
    st.write(f"**User:** `{os.getenv('SNOWFLAKE_USER')}`")
    st.write(f"**Warehouse:** `{os.getenv('SNOWFLAKE_WAREHOUSE')}`")
    if os.getenv("SNOWFLAKE_DATABASE"):
        st.write(f"**Database:** `{os.getenv('SNOWFLAKE_DATABASE')}`")
    if os.getenv("SNOWFLAKE_SCHEMA"):
        st.write(f"**Schema:** `{os.getenv('SNOWFLAKE_SCHEMA')}`")
    else:
        st.write(f"**Schema:** `{DEFAULT_SCHEMA}` (default — exploring all schemas)")

    model_options = list(
        dict.fromkeys(
            [DEFAULT_MODEL, "gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"]
        )
    )
    model = st.selectbox(
        "Model",
        options=model_options,
        index=0,
        help="gpt-4o-mini has a much higher tokens-per-minute limit — prefer it "
        "if you hit rate limits.",
    )

    try:
        mcp_client = get_mcp_client(connection_cache_key())
        st.success(f"Connected · {len(mcp_client.tools)} tools available")
        with st.expander("Available MCP tools"):
            for tool in mcp_client.tools:
                st.markdown(f"- **{tool.name}** — {tool.description}")
    except Exception as exc:  # noqa: BLE001 - show connection errors in UI
        st.error(f"Could not connect to the Snowflake MCP server:\n\n{exc}")
        st.info(
            "Make sure `uv` is installed and your Snowflake credentials are "
            "correct. The first launch downloads the MCP server and may take a "
            "minute."
        )
        st.stop()

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.openai_messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.session_state.chat = []
        st.rerun()

    st.caption("Quick starts")
    quick_prompts = {
        "List databases": "List all databases I can access.",
        "Explore tables": "Show me the tables in the current database and schema.",
        "Summarize data": "Pick an interesting table and show me a sample of its rows.",
    }
    quick_choice = None
    for label, prompt in quick_prompts.items():
        if st.button(label, use_container_width=True):
            quick_choice = prompt

# --- Render existing conversation ----------------------------------------- #
for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        for step in item.get("steps", []):
            render_step(step)
        if item.get("content"):
            st.markdown(item["content"])

# --- Handle new input ------------------------------------------------------ #
user_input = st.chat_input("Ask about your Snowflake data...")
prompt = user_input or quick_choice

if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.openai_messages.append({"role": "user", "content": prompt})
    oai_client = openai.OpenAI()

    with st.chat_message("assistant"):
        steps: list[dict] = []
        try:
            with st.spinner("Thinking..."):
                answer = run_agent(
                    client=oai_client,
                    mcp_client=mcp_client,
                    model=model,
                    messages=st.session_state.openai_messages,
                    on_step=step_recorder(steps),
                )
            st.markdown(answer)
        except openai.RateLimitError as exc:
            answer = (
                "⚠️ OpenAI rate limit hit. Your org's tokens-per-minute (TPM) "
                "limit was exceeded. Try switching to **gpt-4o-mini** in the "
                "sidebar (higher TPM), ask a narrower question, or wait a minute "
                f"and retry.\n\n```\n{exc}\n```"
            )
            st.warning(answer)
        except Exception as exc:  # noqa: BLE001 - keep the app alive on any error
            answer = f"⚠️ Something went wrong while answering:\n\n```\n{exc}\n```"
            st.error(answer)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )
