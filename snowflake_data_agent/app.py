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

import altair as alt
import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent_core import (
    DEFAULT_MODEL,
    DEFAULT_SCHEMA,
    build_server_params,
    build_system_prompt,
    connection_cache_key,
    load_mcp_creds_from_env,
    run_agent,
)
from mcp_client import SnowflakeMCPClient

load_dotenv()

# Open-Issue branding.
OPENISSUE_LOGO_URL = "https://open-issue.com/wp-content/uploads/2025/07/logo.png"
OPENISSUE_ICON_URL = (
    "https://open-issue.com/wp-content/uploads/2026/02/cropped-oi-favi-192x192.png"
)
OPENISSUE_SITE_URL = "https://open-issue.com/"
OPENISSUE_TAGLINE = "making sense of your data℠"

# Per-browser Streamlit session keys (isolated across concurrent users).
_SESSION_KEYS = (
    "app_user",
    "sf_creds",
    "mcp_client",
    "mcp_cache_key",
    "openai_messages",
    "chat",
    "active_schema",
    "last_eval",
    "last_eval_payload",
    "auto_judge",
)


def welcome_message(database: str) -> str:
    """Intro shown at the start of a fresh conversation."""
    database = database or "your Snowflake account"
    return f"""\
👋 **Hi! I'm your Snowflake Data Scientist.**

I'm connected to **`{database}`** and I can help you *explore and make sense of \
your data* — not just run queries, but actively hunt for what's interesting.

**What I can do for you:**
- 🗂️ **Map your data** — list databases, schemas, and tables, and describe any \
table's columns and types.
- 🔎 **Profile tables** — row counts, distinct values, NULL rates, value ranges, \
and category breakdowns.
- 🚨 **Spot anomalies** — outliers, impossible values, duplicates, data-quality \
issues, and surprising distributions.
- 📈 **Find insights** — trends over time, relationships between tables, and \
patterns worth a closer look.
- 📊 **Visualize data** — bar, line, area, scatter, and pie charts, right in \
the chat.
- 💬 **Answer questions** in plain English with read-only `SELECT` queries (I \
never modify your data).

**Try asking:**
- *"What's in this database? Give me the lay of the land."*
- *"Profile the largest table and flag anything that looks off."*
- *"Find data-quality issues or anomalies across the schemas."*
- *"Show me a pie chart of records by category in <table>."*

What would you like to explore? 🔬"""

# --------------------------------------------------------------------------- #
# Connection (per Streamlit browser session — safe for concurrent users)
# --------------------------------------------------------------------------- #


def _close_session_mcp() -> None:
    """Shut down this browser session's MCP client only."""
    client = st.session_state.pop("mcp_client", None)
    st.session_state.pop("mcp_cache_key", None)
    if client is None:
        return
    try:
        client.close()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass


def clear_login_session() -> None:
    """End this user's login: close MCP and wipe session-only state."""
    _close_session_mcp()
    for key in _SESSION_KEYS:
        st.session_state.pop(key, None)


def get_session_mcp_client(creds: dict) -> SnowflakeMCPClient:
    """Return a Snowflake MCP client owned by this browser session.

    Each concurrent Streamlit user gets their own MCP subprocess / Snowflake
    connection. Logout or failure in one session must not tear down others
    (unlike a process-wide ``@st.cache_resource``).
    """
    key = connection_cache_key(creds)
    existing = st.session_state.get("mcp_client")
    if existing is not None and st.session_state.get("mcp_cache_key") == key:
        return existing

    _close_session_mcp()
    client = SnowflakeMCPClient(build_server_params(creds))
    st.session_state.mcp_client = client
    st.session_state.mcp_cache_key = key
    return client


def render_login() -> None:
    """Render the app sign-in form. MCP uses ``.env`` service-account creds."""
    st.image(OPENISSUE_LOGO_URL, width=260)
    st.title("❄️ Snowflake Data Explorer")
    st.caption(
        f"by [Open Issue]({OPENISSUE_SITE_URL}) — *{OPENISSUE_TAGLINE}*  \n"
        "Sign in to explore Snowflake data. Warehouse access uses the MCP "
        "service account from `.env` (key-pair); this login is separate."
    )

    with st.form("login_form"):
        st.subheader("Sign in")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("🔐 Sign in", width="stretch")

    if not submitted:
        return

    if not username.strip() or not password:
        st.error("Please enter a username and password.")
        return

    try:
        st.session_state.sf_creds = load_mcp_creds_from_env()
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state.app_user = username.strip()
    st.rerun()


def build_chart(spec: dict) -> Optional[alt.Chart]:
    """Build an Altair chart from a ``display_chart`` spec. Returns None on failure."""
    try:
        data = spec.get("data") or []
        df = pd.DataFrame(data)
        if df.empty:
            return None

        ctype = (spec.get("chart_type") or "bar").lower()
        x = spec.get("x")
        y = spec.get("y")
        color = spec.get("color")
        title = spec.get("title") or ""

        # Coerce the y field to numeric when the values look numeric.
        if y in df.columns:
            coerced = pd.to_numeric(df[y], errors="coerce")
            if coerced.notna().any():
                df[y] = coerced

        base = alt.Chart(df).properties(title=title, height=380)
        tooltip = list(df.columns)
        color_enc = (
            {"color": alt.Color(f"{color}:N")}
            if color and color in df.columns
            else {}
        )

        if ctype == "pie":
            category = x or color
            if not (category and y):
                return None
            return base.mark_arc().encode(
                theta=alt.Theta(f"{y}:Q"),
                color=alt.Color(f"{category}:N"),
                tooltip=tooltip,
            )

        if not (x and y):
            return None
        if ctype == "bar":
            return base.mark_bar().encode(
                x=alt.X(f"{x}:N", sort="-y"), y=alt.Y(f"{y}:Q"),
                tooltip=tooltip, **color_enc,
            )
        if ctype == "line":
            return base.mark_line(point=True).encode(
                x=alt.X(x), y=alt.Y(f"{y}:Q"), tooltip=tooltip, **color_enc
            )
        if ctype == "area":
            return base.mark_area(opacity=0.6).encode(
                x=alt.X(x), y=alt.Y(f"{y}:Q"), tooltip=tooltip, **color_enc
            )
        if ctype == "scatter":
            return base.mark_circle(size=70).encode(
                x=alt.X(x), y=alt.Y(f"{y}:Q"), tooltip=tooltip, **color_enc
            )
        # Fallback to a bar chart for unknown types.
        return base.mark_bar().encode(
            x=alt.X(f"{x}:N"), y=alt.Y(f"{y}:Q"), tooltip=tooltip, **color_enc
        )
    except Exception:  # noqa: BLE001 - charting is best-effort
        return None

# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def render_chart_step(step: dict) -> None:
    """Render a chart produced by the ``display_chart`` tool, inline (expanded)."""
    spec = step["chart"]
    title = spec.get("title") or "Chart"
    st.markdown(f"**📊 {title}**")
    chart = build_chart(spec)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    else:
        st.warning("Could not render the chart from the provided data.")
    data = spec.get("data") or []
    if data:
        with st.expander("Chart data", expanded=False):
            st.dataframe(pd.DataFrame(data), width="stretch")


def render_step(step: dict) -> None:
    """Render a single agent step (a tool call or a chart) in the UI."""
    if "chart" in step:
        render_chart_step(step)
        return

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
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.caption("Result")
            st.code(step.get("text", ""), language="json")


def render_response_copy(content: str) -> None:
    """Render a copyable Markdown version of a response."""
    with st.expander("Copy response"):
        st.caption("Use the copy button in the code block below.")
        st.code(content, language="markdown")


def render_assistant_response(content: str) -> None:
    """Render assistant text with an easy copy affordance."""
    st.markdown(content)
    render_response_copy(content)


def step_recorder(store: list[dict]):
    """Build an ``on_step`` callback that records steps and live-renders charts.

    Steps are always recorded; only chart steps are rendered (raw MCP tool calls
    stay hidden from the conversation).
    """

    def _on_step(step: dict[str, Any]):
        store.append(step)
        if "chart" in step:
            render_step(step)

    return _on_step


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def prior_chat_tool_steps(chat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect tool steps from earlier assistant turns (exclude the latest)."""
    if len(chat) < 2:
        return []
    prior: list[dict[str, Any]] = []
    for item in chat[:-1]:
        if item.get("role") == "assistant":
            prior.extend(item.get("steps") or [])
    return prior


def score_last_turn(
    *,
    client: openai.OpenAI,
    question: str,
    answer: str,
    steps: list[dict],
    creds: dict,
    active: Optional[dict],
    judge: bool,
    prior_steps: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Score one chat turn with the eval harness metrics."""
    from evals.metrics import score_trace

    return score_trace(
        client,
        question=question,
        answer=answer,
        steps=steps,
        judge=judge,
        creds=creds,
        active=active,
        prior_steps=prior_steps,
    )


def render_eval_sidebar(eval_result: Optional[dict[str, Any]]) -> None:
    """Render last-turn tool stats + optional RAG Triad scores in the sidebar."""
    st.divider()
    st.subheader("Last-turn eval")
    st.checkbox(
        "Auto-score with LLM judges",
        value=False,
        help="Runs RAG Triad + efficiency judges after each answer "
        "(extra OpenAI calls). Tool counts always update.",
        key="auto_judge",
    )
    if not eval_result:
        st.caption("Ask a question to see tool counts and optional triad scores.")
        return

    if eval_result.get("error"):
        st.warning(f"Eval error: {eval_result['error']}")

    eff = eval_result.get("expectations", {}).get("efficiency") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Tools", eff.get("tool_call_count", 0))
    c2.metric("Errors", eff.get("error_count", 0))
    c3.metric("Charts", eff.get("chart_count", 0))
    tools = eff.get("unique_tools") or []
    if tools:
        st.caption("Tools used: `" + "`, `".join(tools) + "`")
    elif (eff.get("tool_call_count") or 0) == 0:
        prior_n = int(eval_result.get("prior_tool_steps") or 0)
        st.warning(
            "This turn called **0 tools**. Context relevance stays 0 "
            "(nothing newly retrieved). "
            + (
                f"Groundedness can still use {prior_n} earlier tool step(s)."
                if prior_n
                else "No earlier tool evidence either — expect groundedness 0."
            )
        )

    triad = eval_result.get("rag_triad_mean")
    if triad is None:
        payload = st.session_state.get("last_eval_payload")
        if payload and st.button(
            "Run LLM judges on last turn",
            width="stretch",
            help="Scores the previous answer with RAG Triad judges.",
        ):
            try:
                with st.spinner("Scoring..."):
                    st.session_state.last_eval = score_last_turn(
                        client=openai.OpenAI(),
                        question=payload["question"],
                        answer=payload["answer"],
                        steps=payload["steps"],
                        creds=payload["creds"],
                        active=payload.get("active"),
                        judge=True,
                        prior_steps=payload.get("prior_steps") or [],
                    )
            except Exception as exc:  # noqa: BLE001
                st.session_state.last_eval = {
                    **eval_result,
                    "error": str(exc),
                }
            st.rerun()
        elif not st.session_state.get("auto_judge"):
            st.caption(
                "Enable auto-score, or click the button above after a turn."
            )
        return

    st.metric("RAG triad mean", _fmt_score(triad))
    rows = []
    for key, label in (
        ("answer_relevance", "Answer relevance"),
        ("groundedness", "Groundedness"),
        ("context_relevance", "Context relevance"),
        ("execution_efficiency", "Execution efficiency"),
    ):
        metric = eval_result.get(key) or {}
        rows.append(
            {
                "Metric": label,
                "Score": _fmt_score(metric.get("score")),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("Judge reasons", expanded=False):
        for key, label in (
            ("answer_relevance", "Answer relevance"),
            ("groundedness", "Groundedness"),
            ("context_relevance", "Context relevance"),
            ("execution_efficiency", "Execution efficiency"),
        ):
            metric = eval_result.get(key) or {}
            reason = (metric.get("reason") or "").strip()
            if reason:
                st.markdown(f"**{label}** ({_fmt_score(metric.get('score'))})")
                st.write(reason)


st.set_page_config(
    page_title="Snowflake Data Explorer · Open Issue",
    page_icon=OPENISSUE_ICON_URL,
    layout="wide",
)

# App-wide logo (top-left of the app and at the top of the sidebar).
try:
    st.logo(
        OPENISSUE_LOGO_URL,
        size="large",
        link=OPENISSUE_SITE_URL,
        icon_image=OPENISSUE_ICON_URL,
    )
except Exception:  # noqa: BLE001 - st.logo is best-effort branding
    pass

# --- Login gate ----------------------------------------------------------- #
if "app_user" not in st.session_state or "sf_creds" not in st.session_state:
    render_login()
    st.stop()

creds = st.session_state.sf_creds
if not creds.get("private_key_file"):
    st.warning(
        "MCP private key missing. Set `SNOWFLAKE_PRIVATE_KEY_FILE` in `.env`."
    )
    clear_login_session()
    render_login()
    st.stop()

st.title("❄️ Snowflake Data Explorer")
st.caption(
    f"Chat with your Snowflake data through an MCP server · "
    f"by [Open Issue]({OPENISSUE_SITE_URL}) — *{OPENISSUE_TAGLINE}*"
)

if "active_schema" not in st.session_state:
    st.session_state.active_schema = None  # {"database":..., "schema":...} or None
if "openai_messages" not in st.session_state:
    st.session_state.openai_messages = [
        {
            "role": "system",
            "content": build_system_prompt(creds, st.session_state.active_schema),
        }
    ]
if "chat" not in st.session_state:
    st.session_state.chat = []  # Display items: {role, content, steps}
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None
if "auto_judge" not in st.session_state:
    st.session_state.auto_judge = False


def apply_active_schema(database: str, schema: str) -> str:
    """Set the conversation's active schema and refresh the system message.

    Updating the system message (index 0, kept by ``trim_messages``) makes the
    active context deterministic and resilient to history trimming.
    """
    st.session_state.active_schema = {"database": database, "schema": schema}
    st.session_state.openai_messages[0]["content"] = build_system_prompt(
        creds, st.session_state.active_schema
    )
    return f"Active context set to {database}.{schema}."

# --- Sidebar: connection + tools ----------------------------------------- #
with st.sidebar:
    st.header("Connection")
    st.write(f"**Signed in:** `{st.session_state.app_user}`")
    st.caption("MCP service account (from `.env`)")
    st.write(f"**Account:** `{creds['account']}`")
    st.write(f"**MCP user:** `{creds['user']}`")
    st.write(f"**Warehouse:** `{creds['warehouse']}`")
    if creds.get("role"):
        st.write(f"**Role:** `{creds['role']}`")
    st.write(f"**Database:** `{creds['database']}`")
    if creds.get("schema"):
        st.write(f"**Schema:** `{creds['schema']}`")
    else:
        st.write(f"**Schema:** `{DEFAULT_SCHEMA}` (default — exploring all schemas)")

    active = st.session_state.active_schema
    if active:
        st.info(f"📌 Active context: `{active['database']}.{active['schema']}`")
        if st.button("↩︎ Use session default", width="stretch"):
            st.session_state.active_schema = None
            st.session_state.openai_messages[0]["content"] = build_system_prompt(
                creds, None
            )
            st.rerun()
    else:
        st.caption("📌 Active context: session default")

    if st.button("🔓 Log out", width="stretch"):
        clear_login_session()
        st.rerun()

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
        with st.spinner("Connecting to Snowflake..."):
            mcp_client = get_session_mcp_client(creds)
        st.success("Connected to Snowflake")
    except Exception as exc:  # noqa: BLE001 - show connection errors in UI
        st.error(f"Could not connect to the Snowflake MCP server:\n\n{exc}")
        st.info(
            "Check your Snowflake credentials in `.env` and try again. Use "
            "**Log out** to return to the sign-in screen. (The first launch also "
            "downloads the MCP server via `uv`, which can take a minute.)"
        )
        # Drop only this session's failed client so retry can reconnect.
        _close_session_mcp()
        st.stop()

    st.divider()
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.active_schema = None
        st.session_state.openai_messages = [
            {"role": "system", "content": build_system_prompt(creds, None)}
        ]
        st.session_state.chat = []
        st.session_state.last_eval = None
        st.session_state.pop("last_eval_payload", None)
        st.rerun()

    st.caption("Quick starts")

    if creds.get("schema"):
        deep_dive_target = f"the {creds['database']}.{creds['schema']} schema"
    else:
        deep_dive_target = (
            f"the {creds['database']} database (first list its schemas, then pick "
            "the most substantial/interesting one to deep-dive)"
        )
    deep_dive_prompt = (
        f"Do a thorough, end-to-end data-science deep-dive of {deep_dive_target}. "
        "Inventory the tables and their sizes; for the most important tables, "
        "infer the grain and profile their columns (counts, distinct values, NULL "
        "rates, ranges, top categories); assess data quality; surface outliers "
        "and anomalies; examine relationships and any trends over time; include "
        "charts for the most important findings; and finish with a "
        "severity-ranked executive summary and recommended next steps."
    )

    quick_prompts = {
        "🔬 Deep-dive this schema": deep_dive_prompt,
        "List databases": "List all databases I can access.",
        "Explore tables": "Show me the tables in the current database and schema.",
        "Summarize data": "Pick an interesting table and show me a sample of its rows.",
        "Chart it": "Pick an interesting table, aggregate a categorical column, "
        "and show me a bar or pie chart of the result.",
    }
    quick_choice = None
    for label, prompt in quick_prompts.items():
        if st.button(label, width="stretch"):
            quick_choice = prompt

    render_eval_sidebar(st.session_state.last_eval)

# --- Render existing conversation ----------------------------------------- #
if not st.session_state.chat:
    with st.chat_message("assistant"):
        render_assistant_response(welcome_message(creds["database"]))

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        for step in item.get("steps", []):
            # Only charts are shown; raw MCP tool calls stay hidden.
            if "chart" in step:
                render_step(step)
        if item.get("content"):
            if item["role"] == "assistant":
                render_assistant_response(item["content"])
            else:
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
                    apply_active_schema=apply_active_schema,
                )
            render_assistant_response(answer)
        except openai.RateLimitError as exc:
            answer = (
                "⚠️ OpenAI rate limit hit. Your org's tokens-per-minute (TPM) "
                "limit was exceeded. Try switching to **gpt-4o-mini** in the "
                "sidebar (higher TPM), ask a narrower question, or wait a minute "
                f"and retry.\n\n```\n{exc}\n```"
            )
            st.warning(answer)
            render_response_copy(answer)
        except Exception as exc:  # noqa: BLE001 - keep the app alive on any error
            answer = f"⚠️ Something went wrong while answering:\n\n```\n{exc}\n```"
            st.error(answer)
            render_response_copy(answer)

    st.session_state.chat.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )
    prior_steps = prior_chat_tool_steps(st.session_state.chat)

    # Always record tool stats; optionally run LLM judges (sidebar toggle).
    try:
        judge = bool(st.session_state.get("auto_judge", False))
        with st.spinner(
            "Scoring last turn..." if judge else "Recording tool stats..."
        ):
            st.session_state.last_eval = score_last_turn(
                client=oai_client,
                question=prompt,
                answer=answer,
                steps=steps,
                creds=creds,
                active=st.session_state.active_schema,
                judge=judge,
                prior_steps=prior_steps,
            )
        st.session_state.last_eval_payload = {
            "question": prompt,
            "answer": answer,
            "steps": steps,
            "prior_steps": prior_steps,
            "creds": creds,
            "active": st.session_state.active_schema,
        }
    except Exception as exc:  # noqa: BLE001 - evals must not break chat
        st.session_state.last_eval = {
            "expectations": {
                "efficiency": {
                    "tool_call_count": len(steps),
                    "error_count": sum(1 for s in steps if s.get("error")),
                    "chart_count": sum(1 for s in steps if "chart" in s),
                    "unique_tools": sorted(
                        {s.get("tool") for s in steps if s.get("tool")}
                    ),
                }
            },
            "prior_tool_steps": len(prior_steps),
            "error": str(exc),
        }
        st.session_state.last_eval_payload = {
            "question": prompt,
            "answer": answer,
            "steps": steps,
            "prior_steps": prior_steps,
            "creds": creds,
            "active": st.session_state.active_schema,
        }
    st.rerun()
