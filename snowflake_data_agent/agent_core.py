"""Core Snowflake data-agent logic (no Streamlit UI).

Shared by ``app.py`` and the headless ``evals/`` harness.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Optional

import openai
from dotenv import load_dotenv
from mcp import StdioServerParameters

from mcp_client import SnowflakeMCPClient

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Max model<->tool round-trips per user turn. Each round can issue several tool
# calls in parallel, so a thorough multi-stage exploration needs headroom.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "18"))

# Token/size guardrails to keep requests under the model's context limit.
# Roughly 4 characters per token.
# - MAX_TOOL_RESULT_CHARS caps what each tool result contributes to the MODEL's
#   context. The UI still renders the FULL result (from step["rows"]/["text"]),
#   so this never reduces what you see — only what the model re-reads each step.
# - MAX_CONTEXT_CHARS is the overall budget for the message history per request;
#   trim_messages() enforces it by dropping old turns and, if needed, shrinking
#   the oldest tool outputs within the current turn.
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "10000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "120000"))  # ~30k tokens

# The MCP server requires *some* default schema to connect. When the user does
# not pin one, fall back to INFORMATION_SCHEMA (present in every database) so the
# agent is free to explore all schemas rather than being scoped to one.
DEFAULT_SCHEMA = "INFORMATION_SCHEMA"

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
  * Prefer Snowflake metadata views (INFORMATION_SCHEMA.TABLES, .COLUMNS) \
to learn structure, row counts, and sizes WITHOUT scanning data.
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
- Do NOT rely on earlier chat memory alone when naming tables, columns, or \
proposing schemas (star schemas, GL inventories, summaries). Even if you listed \
objects in a previous turn, re-call list_tables / describe_table / read_query in \
the CURRENT turn so the answer is freshly grounded. Memory without a tool call \
this turn is treated as ungrounded.
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
- When you name tables/columns in the answer, prefer fully-qualified \
DATABASE.SCHEMA.TABLE (and TABLE.COLUMN) so the user can see exactly which \
object you used — even if the active schema is already set.

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
- Do NOT use Snowflake reserved keywords as column aliases or identifiers \
(e.g. ROWS, VALUES, ORDER, GROUP, TABLE, CURRENT, LOCALTIME, NUMBER). Pick a \
safe descriptive alias instead (e.g. ROW_COUNT, not ROWS) or double-quote it \
("ROWS"). Prefer clear non-reserved aliases like ROW_COUNT, NULL_COUNT, \
DISTINCT_COUNT.
- STATIC SQL ONLY — NO DYNAMIC IDENTIFIERS. You write one static query at a \
time; you cannot build identifiers from strings. NEVER concatenate a column or \
table name into SQL (e.g. t."' || COLUMN_NAME || '" or 'SELECT ' || col) — \
Snowflake parses that text literally and fails with "invalid identifier". To \
reference a column you MUST write its literal name in the query text. You also \
cannot loop/iterate inside SQL. If you need column names, get them first \
(describe_table or INFORMATION_SCHEMA.COLUMNS), then write a query with those \
names spelled out explicitly.
- DON'T GUESS METADATA VIEWS/COLUMNS — VERIFY. Use the correct sources: table \
sizes/row counts come from INFORMATION_SCHEMA.TABLES (columns ROW_COUNT, BYTES, \
TABLE_TYPE); columns from INFORMATION_SCHEMA.COLUMNS. Do not invent views like \
TABLE_STORAGE_METRICS or assume a column exists. If you are unsure of a view's \
columns, inspect it first (SELECT * FROM <db>.INFORMATION_SCHEMA.<view> LIMIT 1). \
An "invalid identifier" error means the column/view NAME is wrong — inspect and \
correct it; do not retry the same query.
- NEVER reference a column you have not confirmed exists. Column names are NOT \
predictable from naming patterns — do not assume a table has ID, NAME, CREATED_AT, \
MI_ID, etc. BEFORE writing any query that names specific columns of a table, get \
its real columns via describe_table('db.schema.table') (or SELECT COLUMN_NAME FROM \
db.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=... AND TABLE_NAME=...). Use ONLY \
the exact names it returns. An "invalid identifier '<COL>'" error means you used a \
name that isn't in the table — re-describe it and correct, never retry blindly.
- PROFILE ONE TABLE PER QUERY with explicit columns. AFTER confirming the column \
list (above), write literal per-column aggregates for that table, e.g.: \
SELECT COUNT(*) AS ROW_COUNT, \
COUNT_IF("ColA" IS NULL) AS COLA_NULLS, COUNT(DISTINCT "ColA") AS COLA_DISTINCT, \
COUNT_IF("ColB" IS NULL) AS COLB_NULLS, COUNT(DISTINCT "ColB") AS COLB_DISTINCT \
FROM DB.SCHEMA.TABLE. Do NOT try to profile many tables and many columns in one \
giant UNION/CASE query — issue separate per-table queries (in parallel tool \
calls when possible) instead.
- Boolean profiling in Snowflake: do NOT write `col IS TRUE` or `col IS FALSE` \
inside COUNT_IF; Snowflake can reject that syntax. Use equality for boolean \
values and IS NULL only for nulls: COUNT_IF(col = TRUE) AS COL_TRUE, \
COUNT_IF(col = FALSE) AS COL_FALSE, COUNT_IF(col IS NULL) AS COL_NULLS. If the \
column name is case-sensitive or special, quote it: COUNT_IF("IS_MANAGER" = TRUE).
- RESULT SERIALIZATION: the MCP server returns rows as JSON and CANNOT serialize \
DATE/TIME/TIMESTAMP, VARIANT/OBJECT/ARRAY, or BINARY values — a query that \
returns them fails with "Object of type Timestamp is not JSON serializable" and \
no rows. So in the SELECT you return, CAST every temporal/semi-structured column \
to text: TO_VARCHAR(col) or col::VARCHAR (e.g. TO_VARCHAR(MIN(ts)) AS MIN_TS, \
TO_VARCHAR(DATE_TRUNC('day', ts)) AS DAY). Return only strings, numbers, and \
booleans. (You may still use DATE/TIMESTAMP in WHERE/GROUP BY/ORDER BY — just \
don't return the raw value in the SELECT list.)
- Parsing text dates/timestamps/numbers — follow this sequence, do NOT skip step 1:
  1. SAMPLE FIRST. Before any conversion, look at real values: \
SELECT DISTINCT col FROM <table> WHERE col IS NOT NULL LIMIT 10. Infer the exact \
format from what you see (e.g. '20260601 062853.000' -> 'YYYYMMDD HH24MISS.FF3').
  2. CONVERT with the TRY_ variant AND an explicit format string \
(TRY_TO_TIMESTAMP(col, 'YYYYMMDD HH24MISS.FF3'), TRY_TO_DATE, TRY_TO_NUMBER, \
TRY_TO_DECIMAL). Never use plain TO_DATE/TO_TIMESTAMP/TO_NUMBER on raw text \
(they abort the query on a single bad value).
  3. VALIDATE the parse. Compare parsed-non-null vs total \
(COUNT_IF(TRY_TO_TIMESTAMP(col, fmt) IS NOT NULL) vs COUNT_IF(col IS NOT NULL)). \
Beware the silent trap: TRY_ returns NULL for BOTH true NULLs AND \
format-mismatches, so a wrong format makes everything NULL. If the parsed \
non-null rate is unexpectedly low, your FORMAT is wrong (not the data) — \
re-sample, try another format, and only report genuine unparseable values once \
the format is confirmed correct.
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


def load_mcp_creds_from_env() -> dict[str, str]:
    """Load Snowflake MCP service-account credentials from the environment.

    These drive the MCP server (key-pair / JWT). They are separate from the
    Streamlit login username/password entered on the app form.
    """
    required = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
        "user": os.getenv("SNOWFLAKE_USER", ""),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", ""),
        "database": os.getenv("SNOWFLAKE_DATABASE", ""),
        "private_key_file": os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
        or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", ""),
    }
    missing = [k for k, v in required.items() if not str(v).strip()]
    if missing:
        raise ValueError(
            "Missing Snowflake MCP env: "
            + ", ".join(missing)
            + ". Set them in .env (service account for the MCP connection)."
        )
    return {
        **{k: str(v).strip() for k, v in required.items()},
        "role": (os.getenv("SNOWFLAKE_ROLE") or "").strip(),
        "schema": (os.getenv("SNOWFLAKE_SCHEMA") or "").strip(),
        "private_key_pwd": os.getenv("SNOWFLAKE_PRIVATE_KEY_PWD") or "",
    }


def build_server_params(creds: dict) -> StdioServerParameters:
    """Build MCP server launch parameters from a credentials dict.

    ``creds`` is expected to contain ``account``, ``user``, ``warehouse`` and
    ``database`` (validated by the login form), plus optional auth fields.
    Key-pair auth uses ``authenticator=snowflake_jwt`` and a private key file;
    Snowflake stores the matching public key on the user as ``RSA_PUBLIC_KEY``.
    """
    # SNOWFLAKE schema is optional: when omitted we default to INFORMATION_SCHEMA
    # so the agent can explore every schema in the database freely.
    schema = creds.get("schema") or DEFAULT_SCHEMA

    command = os.getenv("SNOWFLAKE_MCP_COMMAND", "uv")
    package = os.getenv("SNOWFLAKE_MCP_PACKAGE", "mcp_snowflake_server")

    # We launch the server through ``mcp_server_launcher.py`` instead of the
    # package's console entry point. The launcher monkeypatches ``json.dumps``
    # so the server can serialize Timestamps/dates/Decimals/bytes (a known
    # upstream bug: "Object of type Timestamp is not JSON serializable").
    launcher = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "mcp_server_launcher.py")

    args: list[str] = []
    # ``uv run`` builds an ephemeral environment containing the package. The
    # version pins work around a pyOpenSSL/cryptography incompatibility
    # (missing X509_V_FLAG_NOTIFY_POLICY).
    if os.path.basename(command) in ("uv", "uvx"):
        args += ["run", "--no-project", "--with", package]
        pins = os.getenv("SNOWFLAKE_MCP_UV_WITH", "cryptography<44,setuptools<75")
        for spec in (s.strip() for s in pins.split(",")):
            if spec:
                args += ["--with", spec]
        args += ["python", launcher]
    else:
        # A custom command (e.g. a venv python) runs the launcher directly.
        args += [launcher]

    args += ["--account", creds["account"], "--user", creds["user"]]
    args += ["--warehouse", creds["warehouse"]]
    if creds.get("role"):
        args += ["--role", creds["role"]]
    args += ["--database", creds["database"], "--schema", schema]

    # --- Key-pair authentication (handled by the Snowflake connector) --- #
    # The MCP server forwards any extra ``--key value`` pairs straight to
    # snowflake.connector, so we can use JWT auth without modifying the server.
    # Snowflake stores the matching RSA_PUBLIC_KEY on the user.
    args += ["--authenticator", "snowflake_jwt"]
    private_key_file = (
        creds.get("private_key_file")
        or os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
        or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
    )
    if private_key_file:
        args += ["--private_key_file", private_key_file]
    pk_passphrase = creds.get("private_key_pwd") or os.getenv(
        "SNOWFLAKE_PRIVATE_KEY_PWD"
    )
    if pk_passphrase:
        args += ["--private_key_file_pwd", pk_passphrase]

    # Escape hatch: arbitrary connector params as "key=value,key2=value2".
    extra = os.getenv("SNOWFLAKE_CONNECT_EXTRA", "")
    for pair in (p.strip() for p in extra.split(",")):
        if pair and "=" in pair:
            key, value = pair.split("=", 1)
            args += [f"--{key.strip()}", value.strip()]

    # Pass through the parent environment so ``uv``/PATH resolve correctly.
    env = {k: v for k, v in os.environ.items()}
    return StdioServerParameters(command=command, args=args, env=env)


def connection_cache_key(creds: dict) -> str:
    """A stable key that changes when connection-relevant settings change."""
    key_hash = hashlib.sha256(
        (creds.get("private_key_file", "") + creds.get("private_key_pwd", "")).encode()
    ).hexdigest()[:12]
    parts = [
        creds.get("account", ""),
        creds.get("user", ""),
        creds.get("warehouse", ""),
        creds.get("role", ""),
        creds.get("database", ""),
        creds.get("schema", ""),
        key_hash,
        os.getenv("SNOWFLAKE_MCP_PACKAGE", "mcp_snowflake_server"),
    ]
    return "|".join(parts)


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

    kept = system + body[cut:]

    # Safety net: a single turn can still exceed the budget (many large tool
    # results before any new user message). Shrink the OLDEST tool outputs —
    # without removing messages — so tool_call/tool pairing stays intact. We
    # leave the most recent messages untouched so the model sees fresh results.
    if total > budget:
        protected = 4
        for i in range(max(0, len(kept) - protected)):
            if total <= budget:
                break
            msg = kept[i]
            content = msg.get("content") or ""
            if msg.get("role") == "tool" and len(content) > 600:
                shrunk = dict(msg)
                shrunk["content"] = (
                    content[:600]
                    + "\n... [older tool result truncated to fit the context "
                    "window] ..."
                )
                kept[i] = shrunk
                total -= len(content) - len(shrunk["content"])

    return kept

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


def _truncate_for_model(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Cap a tool result's contribution to the MODEL's context.

    The full result is still shown in the UI (rendered from step["rows"]/["text"]);
    this only limits what is re-sent to the model on each agent step.
    """
    if text is None or len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n... [result truncated to {limit} chars for the model; the full "
        "result is shown in the app. Aggregate or add a smaller LIMIT if you need "
        "more of it in-context]"
    )


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
                    "content": _truncate_for_model(result_text),
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

