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
# Connection
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Connecting to Snowflake...")
def get_mcp_client(cache_key: str, _creds: dict) -> SnowflakeMCPClient:
    """Create (and cache) the MCP client for the given credentials.

    ``cache_key`` is hashed by Streamlit to key the cache; ``_creds`` is prefixed
    with an underscore so Streamlit does not try to hash the (mutable) dict.
    """
    return SnowflakeMCPClient(build_server_params(_creds))

def render_login() -> None:
    """Render the Snowflake sign-in form. Stores creds in session on submit."""
    st.image(OPENISSUE_LOGO_URL, width=260)
    st.title("❄️ Snowflake Data Explorer")
    st.caption(
        f"by [Open Issue]({OPENISSUE_SITE_URL}) — *{OPENISSUE_TAGLINE}*  \n"
        "Sign in to your Snowflake instance to start exploring your data."
    )

    with st.form("login_form"):
        st.subheader("Connect to Snowflake")
        account = st.text_input(
            "Account identifier",
            value=os.getenv("SNOWFLAKE_ACCOUNT", ""),
            help="e.g. `orgname-accountname` or `locator.region` "
            "(from your Snowsight URL).",
        )
        user = st.text_input("User", value=os.getenv("SNOWFLAKE_USER", ""))

        col_w, col_d = st.columns(2)
        with col_w:
            warehouse = st.text_input(
                "Warehouse", value=os.getenv("SNOWFLAKE_WAREHOUSE", "")
            )
        with col_d:
            database = st.text_input(
                "Database", value=os.getenv("SNOWFLAKE_DATABASE", "")
            )

        col_r, col_s = st.columns(2)
        with col_r:
            role = st.text_input(
                "Role (optional)", value=os.getenv("SNOWFLAKE_ROLE", "")
            )
        with col_s:
            schema = st.text_input(
                "Schema (optional)",
                value=os.getenv("SNOWFLAKE_SCHEMA", ""),
                placeholder=f"{DEFAULT_SCHEMA} (explore all schemas)",
            )

        with st.expander("Key-pair authentication"):
            st.caption(
                "The app uses Snowflake key-pair auth (`snowflake_jwt`). "
                "Register the matching public key on the Snowflake user as "
                "`RSA_PUBLIC_KEY`, then provide the private `.p8` file here."
            )
            private_key_file = st.text_input(
                "Private key file",
                value=os.getenv(
                    "SNOWFLAKE_PRIVATE_KEY_FILE",
                    os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", ""),
                ),
                placeholder="/Users/thongbui/.snowflake/rsa_key.p8",
                help="Snowflake stores the matching RSA_PUBLIC_KEY on the user; "
                "the app authenticates with this private key file.",
            )
            private_key_pwd = st.text_input(
                "Private key passphrase (optional)",
                type="password",
                value=os.getenv("SNOWFLAKE_PRIVATE_KEY_PWD", ""),
            )

        submitted = st.form_submit_button("🔐 Connect", width="stretch")

    if submitted:
        fields = {
            "Account": account,
            "User": user,
            "Warehouse": warehouse,
            "Database": database,
            "Private key file": private_key_file,
        }
        missing = [name for name, value in fields.items() if not value.strip()]
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
            return
        st.session_state.sf_creds = {
            "account": account.strip(),
            "user": user.strip(),
            "warehouse": warehouse.strip(),
            "role": role.strip(),
            "database": database.strip(),
            "schema": schema.strip(),
            "private_key_file": private_key_file.strip(),
            "private_key_pwd": private_key_pwd,
        }
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
if "sf_creds" not in st.session_state:
    render_login()
    st.stop()

creds = st.session_state.sf_creds
if not creds.get("private_key_file"):
    st.warning("Please reconnect with a Snowflake private key file.")
    for key in ("sf_creds", "openai_messages", "chat", "active_schema"):
        st.session_state.pop(key, None)
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
    st.write(f"**Account:** `{creds['account']}`")
    st.write(f"**User:** `{creds['user']}`")
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
        get_mcp_client.clear()  # drop cached connection(s)
        for key in ("sf_creds", "openai_messages", "chat", "active_schema"):
            st.session_state.pop(key, None)
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
        mcp_client = get_mcp_client(connection_cache_key(creds), creds)
        st.success("Connected to Snowflake")
    except Exception as exc:  # noqa: BLE001 - show connection errors in UI
        st.error(f"Could not connect to the Snowflake MCP server:\n\n{exc}")
        st.info(
            "Check your Snowflake credentials and try again. Use **Log out** to "
            "return to the sign-in screen. (The first launch also downloads the "
            "MCP server via `uv`, which can take a minute.)"
        )
        # Drop the failed cached client so the next attempt reconnects cleanly.
        get_mcp_client.clear()
        st.stop()

    st.divider()
    if st.button("🗑️ Clear conversation", width="stretch"):
        st.session_state.active_schema = None
        st.session_state.openai_messages = [
            {"role": "system", "content": build_system_prompt(creds, None)}
        ]
        st.session_state.chat = []
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
