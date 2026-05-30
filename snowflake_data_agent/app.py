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

import hashlib
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

# Token/size guardrail to stay under the OpenAI tokens-per-minute (TPM) limit.
# Tool results are passed back to the model in full; we only trim *older* turns
# from the history to keep requests bounded. Roughly 4 characters per token.
MAX_CONTEXT_CHARS = 48000  # ~12k tokens of history per request (well under 30k)

# The MCP server requires *some* default schema to connect. When the user does
# not pin one, fall back to INFORMATION_SCHEMA (present in every database) so the
# agent is free to explore all schemas rather than being scoped to one.
DEFAULT_SCHEMA = "INFORMATION_SCHEMA"

# Open-Issue branding.
OPENISSUE_LOGO_URL = "https://open-issue.com/wp-content/uploads/2025/07/logo.png"
OPENISSUE_ICON_URL = (
    "https://open-issue.com/wp-content/uploads/2026/02/cropped-oi-favi-192x192.png"
)
OPENISSUE_SITE_URL = "https://open-issue.com/"
OPENISSUE_TAGLINE = "making sense of your data℠"

SYSTEM_PROMPT = """\
You are a senior Data Scientist working inside the user's Snowflake account. \
Your job is not just to fetch rows on request — you proactively explore the \
data, surface what is interesting, and call out anomalies the user may not have \
thought to ask about. Think like an analyst doing exploratory data analysis \
(EDA): profile the data, form hypotheses, and verify them with queries.

You have tools to:
- list databases, schemas, and tables,
- describe a table's columns and types,
- run read-only SELECT queries.

Exploration workflow:
- To explore broadly, start by listing databases, then schemas, then tables.
- The connection has a default database/schema, but it is only a starting \
context. When the user wants to "explore everything", enumerate ALL schemas in \
the database with list_schemas and inspect tables across them — do not limit \
yourself to the default schema.
- Before querying an unfamiliar table, describe it to learn its columns, types, \
and likely grain (what one row represents).
- Profile tables before drawing conclusions: row counts, distinct counts, \
NULL/blank rates, min/max/avg for numerics, date ranges for timestamps, top \
categories for low-cardinality columns. Use SQL aggregates \
(COUNT, COUNT(DISTINCT ...), MIN, MAX, AVG, STDDEV, APPROX_PERCENTILE, \
GROUP BY) rather than pulling raw rows when profiling.

What to look for (anomalies & insights):
- Data quality issues: unexpected NULL rates, empty strings, duplicate keys, \
constant or near-constant columns, mixed formats, encoding/casing \
inconsistencies.
- Outliers and extremes: values far outside typical ranges, negative values \
where only positives make sense, impossible dates (future timestamps, epoch-0), \
suspicious spikes or drops over time.
- Distribution surprises: heavy skew, unexpected gaps, bimodality, dominant \
categories, referential mismatches between related tables.
- Trends and relationships: growth/decline over time, seasonality, correlations \
worth flagging. State these as hypotheses and verify with a follow-up query.

Reporting style:
- Lead with the headline insight or anomaly, then show the supporting evidence \
(a small table or aggregate). Quantify findings ("12% of orders have NULL \
region", not "some orders").
- Clearly separate what the data SHOWS from what you INFER. Note caveats and \
suggest the next query worth running.
- Keep prose concise and skimmable; prefer bullet points and small result sets.

Safety & mechanics:
- ALWAYS add a LIMIT (default 100) to exploratory row-level SELECTs so you never \
pull huge result sets; aggregates can omit LIMIT when naturally bounded.
- Only issue read-only SELECT statements. Never attempt INSERT/UPDATE/DELETE/DDL.
- Qualify object names as DATABASE.SCHEMA.TABLE when the context is ambiguous.
- If a tool returns an error, explain it and suggest how to fix it (e.g. wrong \
database/schema, missing privileges, warehouse not running).
"""


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
- 💬 **Answer questions** in plain English with read-only `SELECT` queries (I \
never modify your data).

**Try asking:**
- *"What's in this database? Give me the lay of the land."*
- *"Profile the largest table and flag anything that looks off."*
- *"Find data-quality issues or anomalies across the schemas."*
- *"Which tables have grown the most recently?"*

What would you like to explore? 🔬"""


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def build_server_params(creds: dict) -> StdioServerParameters:
    """Build MCP server launch parameters from a credentials dict.

    ``creds`` is expected to contain ``account``, ``user``, ``password`` and
    ``warehouse`` and ``database`` (validated by the login form), plus optional
    ``role`` and ``schema``.
    """
    # SNOWFLAKE schema is optional: when omitted we default to INFORMATION_SCHEMA
    # so the agent can explore every schema in the database freely.
    schema = creds.get("schema") or DEFAULT_SCHEMA

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
    args += ["--account", creds["account"], "--user", creds["user"]]
    args += ["--password", creds["password"], "--warehouse", creds["warehouse"]]
    if creds.get("role"):
        args += ["--role", creds["role"]]
    args += ["--database", creds["database"], "--schema", schema]

    # Pass through the parent environment so ``uvx``/PATH resolve correctly.
    env = {k: v for k, v in os.environ.items()}
    return StdioServerParameters(command=command, args=args, env=env)


@st.cache_resource(show_spinner="Connecting to Snowflake via MCP...")
def get_mcp_client(cache_key: str, _creds: dict) -> SnowflakeMCPClient:
    """Create (and cache) the MCP client for the given credentials.

    ``cache_key`` is hashed by Streamlit to key the cache; ``_creds`` is prefixed
    with an underscore so Streamlit does not try to hash the (mutable) dict.
    """
    return SnowflakeMCPClient(build_server_params(_creds))


def connection_cache_key(creds: dict) -> str:
    """A stable key that changes when connection-relevant settings change."""
    pw_hash = hashlib.sha256(creds.get("password", "").encode()).hexdigest()[:12]
    parts = [
        creds.get("account", ""),
        creds.get("user", ""),
        creds.get("warehouse", ""),
        creds.get("role", ""),
        creds.get("database", ""),
        creds.get("schema", ""),
        pw_hash,
        os.getenv("SNOWFLAKE_MCP_PACKAGE", "mcp_snowflake_server"),
    ]
    return "|".join(parts)


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
        col_u, col_p = st.columns(2)
        with col_u:
            user = st.text_input("User", value=os.getenv("SNOWFLAKE_USER", ""))
        with col_p:
            password = st.text_input("Password", type="password")

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

        submitted = st.form_submit_button("🔐 Connect", use_container_width=True)

    if submitted:
        fields = {
            "Account": account,
            "User": user,
            "Password": password,
            "Warehouse": warehouse,
            "Database": database,
        }
        missing = [name for name, value in fields.items() if not value.strip()]
        if missing:
            st.error("Please fill in: " + ", ".join(missing))
            return
        st.session_state.sf_creds = {
            "account": account.strip(),
            "user": user.strip(),
            "password": password,
            "warehouse": warehouse.strip(),
            "role": role.strip(),
            "database": database.strip(),
            "schema": schema.strip(),
        }
        st.rerun()


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
                    "content": result_text,
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

st.title("❄️ Snowflake Data Explorer")
st.caption(
    f"Chat with your Snowflake data through an MCP server · "
    f"by [Open Issue]({OPENISSUE_SITE_URL}) — *{OPENISSUE_TAGLINE}*"
)

if "openai_messages" not in st.session_state:
    st.session_state.openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
if "chat" not in st.session_state:
    st.session_state.chat = []  # Display items: {role, content, steps}

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

    if st.button("🔓 Log out", use_container_width=True):
        get_mcp_client.clear()  # drop cached connection(s)
        for key in ("sf_creds", "openai_messages", "chat"):
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
        st.success(f"Connected · {len(mcp_client.tools)} tools available")
        with st.expander("Available MCP tools"):
            for tool in mcp_client.tools:
                st.markdown(f"- **{tool.name}** — {tool.description}")
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
if not st.session_state.chat:
    with st.chat_message("assistant"):
        st.markdown(welcome_message(creds["database"]))

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
