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

import altair as alt
import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from mcp import StdioServerParameters

from mcp_client import SnowflakeMCPClient

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Max model<->tool round-trips per user turn. Each round can issue several tool
# calls in parallel, so a thorough multi-stage exploration needs headroom.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "18"))

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
You are a world-class Data Scientist embedded in the user's Snowflake account — \
the kind who can be dropped into an unfamiliar schema and, within minutes, \
explain what the data is, how trustworthy it is, and what is genuinely \
interesting about it. You are rigorous, methodical, hypothesis-driven, and \
proactive: you anticipate the next question instead of waiting to be asked.

TOOLS
- list_databases, list_schemas, list_tables — inventory the account.
- describe_table — column names, types, and structure.
- read_query — run read-only SELECTs; this is your primary instrument.
- display_chart — visualize results in the UI.

OPERATING PRINCIPLES
- Be autonomous. When asked to explore a schema or table, carry out the full \
methodology below end-to-end WITHOUT pausing to ask permission for each step. \
Only ask the user when genuinely blocked (ambiguous target, missing access, or \
a destructive request).
- Be hypothesis-driven. State a hypothesis, test it with SQL, and report whether \
it held. Treat every surprise as a lead worth chasing.
- Be efficient with queries, compute, and context. Each query costs money and \
tokens, so:
  * Prefer Snowflake metadata views (INFORMATION_SCHEMA.TABLES, .COLUMNS, \
.TABLE_STORAGE_METRICS) to learn structure, row counts, and sizes WITHOUT \
scanning data.
  * Profile MANY columns in ONE query using conditional aggregates — e.g. \
COUNT(col), COUNT(DISTINCT col), MIN, MAX, AVG, STDDEV, \
SUM(IFF(col IS NULL,1,0)) — rather than one query per column.
  * Issue independent queries in PARALLEL (multiple tool calls in a single turn).
  * On large tables, use approximate functions (APPROX_COUNT_DISTINCT, \
APPROX_PERCENTILE, HLL) and TABLESAMPLE/SAMPLE to estimate fast — and say when a \
number is approximate.
  * Never SELECT * on wide or large tables when profiling; select only what you need.

ACTIVE SCHEMA CONTEXT
- When the user names a specific database and schema (e.g. "STAGING.ANALYTICS", \
"explore the ANALYTICS schema in STAGING", or just "the ANALYTICS schema"), \
treat that DATABASE.SCHEMA as the ACTIVE CONTEXT for the rest of the \
conversation. Interpret later references like "this schema", "the tables", \
"that table", or a bare table name as belonging to the active database.schema — \
until the user explicitly names a different database/schema, at which point you \
switch the active context to that one. When the active schema is set or changes, \
FIRST call the `set_active_schema` tool (so the app tracks it deterministically), \
then briefly confirm it (e.g. "Working in STAGING.ANALYTICS").
- ALWAYS reference tables with FULLY-QUALIFIED three-part names \
(DATABASE.SCHEMA.TABLE) built from the active context. Never use bare two-part \
(SCHEMA.TABLE) names — they get mis-resolved against the connection's default \
database (e.g. `analytics.events` wrongly becomes \
`DEFAULT_DB.ANALYTICS.EVENTS`).
- NEVER invent table or column names. Before querying a table, confirm it exists \
via list_tables for the active schema (and describe_table for its columns). If a \
referenced object is unknown, list the schema's tables first and use the real \
names.
- If it is unclear which schema the user means, use the active context when one \
is set; otherwise fall back to the connection's default database/schema or ask.

EXPLORATION METHODOLOGY (adapt intelligently; you don't have to do every step)
1. Orient. Inventory schemas/tables. Use INFORMATION_SCHEMA to get row counts, \
table type (TABLE/VIEW), created/last_altered, and bytes. Tackle the biggest and \
most central tables first.
2. Understand structure. describe_table, then infer the GRAIN (what one row \
represents), candidate primary keys, and likely foreign keys (by naming/typing). \
Sketch how tables relate.
3. Profile columns, type-aware:
   * Numeric — count, nulls, min/max, avg, stddev, percentiles \
(p01/p25/p50/p75/p99), zeros and negatives.
   * Categorical/low-cardinality — distinct count, top-N values with frequency \
and % share, blanks.
   * Temporal — min/max, range, recency, counts per period, gaps.
   * Text/high-cardinality — length distribution, sample distinct values, format \
consistency.
   * Boolean/flags — true/false balance.
4. Assess data quality. Quantify: null/blank rates, duplicate keys \
(GROUP BY key HAVING COUNT(*) > 1), constant/near-constant columns, mixed \
formats/casing, impossible values (negative amounts, future timestamps, \
epoch-0/1970 dates), and referential-integrity gaps between related tables.
5. Find outliers & distributions. Flag values beyond the IQR fences \
(Q1 - 1.5*IQR, Q3 + 1.5*IQR) or |z| > 3; note skew, bimodality, dominant \
categories, and long tails.
6. Cross-table & relationships. Verify join keys, check cardinality/fan-out, \
find orphans/mismatches, and surface meaningful joins.
7. Trends, segments & correlations. Time series (growth, seasonality, \
spikes/drops), segment comparisons, and relationships between variables \
(CORR, ratios). Correlation is not causation — say so.
8. Synthesize. Pull it together: what the data is, how trustworthy it is, and \
the most interesting/actionable findings.

ANALYTICAL RIGOR
- Define thresholds explicitly (e.g. "outlier = |z| > 3", "near-constant = one \
value in >99% of rows"). Quantify everything with counts AND percentages — never \
vague words like "some" or "a lot".
- Distinguish clearly what the data SHOWS from what you INFER; note assumptions, \
sample sizes, and approximation caveats.
- Sanity-check every number against the table's total row count.

VISUALIZING RESULTS
- Proactively visualize when a chart conveys a finding better than text \
(distributions, top-N, trends, composition) and whenever the user asks.
- Workflow: aggregate with read_query (keep it small), then call display_chart \
with the rows as `data`, the right `chart_type`, `x`, `y`, and a clear `title`. \
Bar = category/top-N comparison; line/area = trend over time; scatter = \
relationship; pie = parts-of-a-whole with few categories. Describe the chart in text.

REPORTING STYLE
- Open with a 1-3 sentence executive summary, then list findings ranked by \
importance/severity (use 🔴 high, 🟡 medium, 🟢 minor/FYI).
- For each finding give: what it is, the evidence (a number, small table, or \
chart), why it matters ("so what"), and a suggested next step.
- Be concise and skimmable — bullets, small result sets, bold the key numbers. \
Close with "Recommended next steps" or open questions worth pursuing.

SAFETY & MECHANICS
- READ-ONLY. Only run SELECT / SHOW / DESCRIBE-style queries. NEVER \
INSERT/UPDATE/DELETE/MERGE/CREATE/DROP/ALTER or anything that mutates data or \
schema.
- Add LIMIT (default 100) to row-level previews; naturally bounded aggregate or \
metadata queries need no LIMIT.
- Qualify object names as DATABASE.SCHEMA.TABLE when ambiguous. The connection \
has a default database/schema, but you may explore ALL schemas in the database.
- Quote identifiers with double quotes when they are case-sensitive or contain \
special characters.
- If a query errors, read the message, explain the likely cause (wrong object, \
insufficient privileges, suspended warehouse, type mismatch), and adjust — do \
NOT repeatedly retry the same failing query.
"""


def build_system_prompt(creds: dict, active: Optional[dict] = None) -> str:
    """Append the current default + active database/schema to the system prompt.

    ``active`` (when set, ``{"database":..., "schema":...}``) is the schema the
    user has steered into during the conversation; it is re-injected every turn
    so it is deterministic and survives history trimming. When ``active`` is not
    set, the agent falls back to the login default below.
    """
    login_db = creds.get("database") or "(unknown)"
    login_schema = creds.get("schema")

    lines = [f"- Login default database: {login_db}"]
    if login_schema:
        lines.append(f"- Login default schema: {login_schema}")
    else:
        lines.append(
            f"- Login default schema: none pinned (connection uses "
            f"{DEFAULT_SCHEMA} for metadata only — not a data schema)."
        )

    if active and active.get("database") and active.get("schema"):
        a_db, a_schema = active["database"], active["schema"]
        lines.append(
            f"- ACTIVE CONTEXT (set during this conversation): {a_db}.{a_schema}. "
            f"When the user does NOT name a database/schema, operate here and "
            f"fully-qualify tables as {a_db}.{a_schema}.TABLE."
        )
    elif login_schema:
        lines.append(
            f"- When the user does NOT name a database/schema, operate in the "
            f"login default {login_db}.{login_schema} and fully-qualify tables as "
            f"{login_db}.{login_schema}.TABLE."
        )
    else:
        lines.append(
            f"- When the user does NOT name a database/schema, stay within the "
            f"{login_db} database; if a single schema is needed and none is "
            f"named, list {login_db}'s schemas and pick/ask — do not query other "
            f"databases unless asked."
        )

    lines.append(
        "- Whenever the user names a (different) database/schema, FIRST call the "
        "set_active_schema tool to update the active context, then proceed. "
        "Keep using the active context for all later turns until it changes."
    )

    return (
        SYSTEM_PROMPT
        + "\n\nSESSION CONTEXT (current Snowflake login)\n"
        + "\n".join(lines)
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


@st.cache_resource(show_spinner="Connecting to Snowflake...")
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


# A client-side tool that lets the model set the conversation's active schema,
# which is tracked deterministically in session state (not just in chat history).
SET_ACTIVE_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "set_active_schema",
        "description": (
            "Set the active database + schema context for the conversation. Call "
            "this whenever the user names or switches to a database/schema, BEFORE "
            "running queries against it. The active context persists across turns "
            "until changed and is used to fully-qualify table names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "database": {"type": "string", "description": "Database name."},
                "schema": {"type": "string", "description": "Schema name."},
            },
            "required": ["database", "schema"],
        },
    },
}


# A client-side tool (not part of the MCP server) that lets the model render a
# chart in the Streamlit UI. The model passes the data points to plot.
DISPLAY_CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "display_chart",
        "description": (
            "Render a chart in the app UI for the user. Call this whenever the "
            "user asks for a chart, graph, plot, pie, or visualization. Provide "
            "the actual data points to plot (typically taken from a prior "
            "read_query result), already aggregated/summarized — keep it small "
            "(ideally < 50 points). After charting, still summarize the chart in "
            "your text answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "area", "scatter", "pie"],
                    "description": "The type of chart to render.",
                },
                "title": {"type": "string", "description": "Chart title."},
                "data": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "description": (
                        "Rows to plot; each row is an object of column -> value."
                    ),
                },
                "x": {
                    "type": "string",
                    "description": (
                        "Field for the x-axis (or the category/label field for a "
                        "pie chart)."
                    ),
                },
                "y": {
                    "type": "string",
                    "description": (
                        "Numeric field for the y-axis (or the slice-size field "
                        "for a pie chart)."
                    ),
                },
                "color": {
                    "type": "string",
                    "description": "Optional field used to group/color series.",
                },
            },
            "required": ["chart_type", "data", "x", "y"],
        },
    },
}


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


def run_agent(
    client: openai.OpenAI,
    mcp_client: SnowflakeMCPClient,
    model: str,
    messages: list[dict],
    on_step,
    apply_active_schema=None,
) -> str:
    """Run the tool-calling loop until the model produces a final answer.

    ``messages`` is the full OpenAI message history (mutated in place).
    ``on_step`` is a callback taking a prebuilt ``step`` dict, used to render
    each tool call (or chart) in the UI.
    ``apply_active_schema(database, schema) -> str`` updates the active-schema
    state and returns a confirmation string (and is expected to update the
    system message in ``messages`` in place).
    Returns the final assistant text.
    """
    tools = mcp_client.openai_tools() + [SET_ACTIVE_SCHEMA_TOOL, DISPLAY_CHART_TOOL]

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

            if name == "set_active_schema":
                # Client-side tool: update the deterministic active-schema state.
                db = arguments.get("database")
                sc = arguments.get("schema")
                if db and sc and apply_active_schema is not None:
                    result_text = apply_active_schema(db, sc)
                else:
                    result_text = "Need both 'database' and 'schema' to set context."
                on_step({"tool": "set_active_schema", "arguments": arguments})
            elif name == "display_chart":
                # Client-side tool: render a chart instead of calling the server.
                point_count = len(arguments.get("data") or [])
                on_step(
                    {
                        "tool": "display_chart",
                        "arguments": {
                            k: v for k, v in arguments.items() if k != "data"
                        },
                        "chart": arguments,
                    }
                )
                result_text = (
                    f"Chart ('{arguments.get('chart_type')}') rendered in the UI "
                    f"with {point_count} data point(s)."
                )
            else:
                try:
                    result = mcp_client.call_tool(name, arguments)
                    result_text = result.text or "(no output)"
                    on_step(
                        {
                            "tool": name,
                            "arguments": arguments,
                            "text": result.text,
                            "rows": result.rows,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - report failures to model
                    result_text = f"Tool call failed: {exc}"
                    on_step({"tool": name, "arguments": arguments, "error": str(exc)})

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
def render_chart_step(step: dict) -> None:
    """Render a chart produced by the ``display_chart`` tool, inline (expanded)."""
    spec = step["chart"]
    title = spec.get("title") or "Chart"
    st.markdown(f"**📊 {title}**")
    chart = build_chart(spec)
    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Could not render the chart from the provided data.")
    data = spec.get("data") or []
    if data:
        with st.expander("Chart data", expanded=False):
            st.dataframe(pd.DataFrame(data), use_container_width=True)


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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("Result")
            st.code(step.get("text", ""), language="json")


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
        if st.button("↩︎ Use session default", use_container_width=True):
            st.session_state.active_schema = None
            st.session_state.openai_messages[0]["content"] = build_system_prompt(
                creds, None
            )
            st.rerun()
    else:
        st.caption("📌 Active context: session default")

    if st.button("🔓 Log out", use_container_width=True):
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
    if st.button("🗑️ Clear conversation", use_container_width=True):
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
        if st.button(label, use_container_width=True):
            quick_choice = prompt

# --- Render existing conversation ----------------------------------------- #
if not st.session_state.chat:
    with st.chat_message("assistant"):
        st.markdown(welcome_message(creds["database"]))

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        for step in item.get("steps", []):
            # Only charts are shown; raw MCP tool calls stay hidden.
            if "chart" in step:
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
                    apply_active_schema=apply_active_schema,
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
