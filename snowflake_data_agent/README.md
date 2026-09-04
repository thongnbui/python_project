# Snowflake Data Explorer (MCP chat agent)

A Streamlit chat app that connects to your Snowflake instance through the
[`mcp_snowflake_server`](https://github.com/isaacwasserman/mcp-snowflake-server)
Model Context Protocol (MCP) server. Ask questions in plain English and an
OpenAI model uses the MCP tools to explore your databases, schemas, tables and
their data.

## How it works

This is a **single-agent** app: one OpenAI model calls tools in a loop until it
can answer. There is no multi-agent planner/executor graph.

```
  ┌─ Login (app.py) ─────────────────────────────────────┐
  │  Snowflake user/password validated against Snowflake │
  │  MCP service account still loaded from .env (JWT)    │
  └──────────────────────────┬───────────────────────────┘
                             │
                             ▼
  ┌─ Streamlit UI (app.py) ──────────────────────────────┐
  │  Chat  ·  charts  ·  sidebar (connection + last-turn │
  │  eval). Each browser = own session + MCP connection. │
  └──────────────────────────┬───────────────────────────┘
                             │ user question
                             ▼
  ┌─ Agent loop (agent_core.run_agent) ──────────────────┐
  │                                                      │
  │   OpenAI model  ◄──────────────────────────────┐     │
  │        │                                       │     │
  │        │ tool calls                            │     │
  │        ▼                                       │     │
  │   ┌────────────────────┐   ┌────────────────┐  │     │
  │   │ Local tools        │   │ MCP tools      │  │     │
  │   │ • set_active_schema│   │ • list_*       │  │     │
  │   │ • display_chart    │   │ • describe_    │  │     │
  │   └─────────┬──────────┘   │   table        │  │     │
  │             │              │ • read_query   │  │     │
  │             │              └───────┬────────┘  │     │
  │             │                      │           │     │
  │             │              mcp_client.py       │     │
  │             │                      │           │     │
  │             │                      ▼           │     │
  │             │         mcp_snowflake_server     │     │
  │             │         (uv + launcher)          │     │
  │             │                      │           │     │
  │             │                      ▼           │     │
  │             │                 Snowflake        │     │
  │             │                      │           │     │
  │             └──────────┬───────────┘           │     │
  │                        │ tool results          │     │
  │                        └───────────────────────┘     │
  │                                                      │
  │   Final answer (+ chart specs) ──► UI                │
  └──────────────────────────┬───────────────────────────┘
                             │ optional
                             ▼
  ┌─ Last-turn eval (evals/metrics.py) ──────────────────┐
  │  Tool counts always · LLM judges when enabled        │
  │  (see diagram below)                                 │
  └──────────────────────────────────────────────────────┘
```

The model is given the MCP server's tools (`list_databases`, `list_schemas`,
`list_tables`, `describe_table`, `read_query`, ...) plus two local tools
(`set_active_schema`, `display_chart`). It calls them in a loop until it can
answer. Charts render inline; raw MCP tool expanders stay hidden in the chat
(steps are still recorded for debugging/evals).

`mcp_client.py` hosts the async MCP session on a background event-loop thread so
it can be driven from Streamlit's synchronous, rerun-on-every-click model.
`mcp_server_launcher.py` patches JSON serialization before starting the MCP
server (upstream Timestamp/Decimal issue).

## Setup

> Run all commands from **inside** the `snowflake_data_agent/` folder:
>
> ```bash
> cd snowflake_data_agent
> ```

1. **Install `uv`** (used to launch the MCP server and, optionally, to create the
   virtual environment). See <https://docs.astral.sh/uv/>. On macOS:
   `brew install uv`.

2. **Create a virtual environment and install dependencies.**

   Using `uv` (fast, recommended):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

   Or using plain Python:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

3. **Configure the OpenAI key.** Copy the example env file and fill it in:

```bash
cp .env.example .env
```

The UI requires `OPENAI_API_KEY`. Default chat model is `gpt-4o-mini`
(override with `OPENAI_MODEL`).

The Streamlit **login page** asks for a Snowflake **username and password**
(and an optional **MFA / TOTP** code) and validates them against Snowflake
using `SNOWFLAKE_ACCOUNT` from `.env`. After a successful login, warehouse
queries still run through a separate **MCP service account** from `.env`
(key-pair / JWT):

```bash
SNOWFLAKE_ACCOUNT="..."
SNOWFLAKE_USER="your_mcp_service_user"   # e.g. S19_INT_AIRFLOW_AUTH_423
SNOWFLAKE_WAREHOUSE="..."
SNOWFLAKE_DATABASE="..."
SNOWFLAKE_PRIVATE_KEY_FILE="/path/to/rsa_key.p8"
# SNOWFLAKE_ROLE / SNOWFLAKE_SCHEMA optional
```

The headless eval harness reads the same `SNOWFLAKE_*` keys.

Generate a private/public key pair for the **MCP** user:

```bash
mkdir -p ~/.snowflake
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/rsa_key.p8 -nocrypt
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub
```

Register the public key on that Snowflake user (paste the body of
`rsa_key.pub`, without the `-----BEGIN PUBLIC KEY-----` / `-----END PUBLIC KEY-----`
lines):

```sql
ALTER USER <mcp_service_user> SET RSA_PUBLIC_KEY='MIIBIjANBgkq...';
```

On the login screen you enter **Username** / **Password**. Leave
`SNOWFLAKE_SCHEMA` blank to explore **all** schemas; it defaults to
`INFORMATION_SCHEMA` just to satisfy the connection. Use **Log out** in the
sidebar to end the app session.

4. **Run the app** using the venv's Streamlit directly:

```bash
.venv/bin/streamlit run app.py
```

The first launch downloads the MCP server via `uv run`, which can take a minute.

## Try asking

- "List all databases I can access."
- "What tables are in the SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 schema?"
- "Describe the CUSTOMER table and show me 10 sample rows."
- "Which 5 nations have the most customers?"
- "Show me a **pie chart** of orders by status." / "Plot a bar chart of revenue by month."

## Evals

A headless golden-query harness lives in `evals/` (RAG Triad-style judges +
tool-trace expectation checks). See [`evals/README.md`](evals/README.md).

```bash
.venv/bin/python -m evals.run_eval --tags smoke --no-judge
.venv/bin/python -m evals.run_eval --app-version "v1: base"
```

In the Streamlit app, the sidebar **Last-turn eval** panel shows tool counts after
each answer. Enable **Auto-score with LLM judges** (or click **Run LLM judges on
last turn**) to see RAG Triad scores for the latest reply.

### How the LLM judge works

The **chat agent** (tool-calling model) and the **judge** (separate OpenAI call,
default `EVAL_JUDGE_MODEL=gpt-4o`) are different models. After a turn finishes,
`score_trace` packs the question, answer, and tool trace, then asks the judge
for JSON scores:

```
  User question
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Agent (tool-calling model)             │
  │  question ──► tools / SQL ──► answer    │
  │               ▲                         │
  │          tool steps (trace)             │
  └──────────────────┬──────────────────────┘
                     │
                     │  question + answer + packed context
                     ▼
  ┌─────────────────────────────────────────┐
  │  score_trace (evals/metrics.py)         │
  │                                         │
  │  1. Expectation checks (deterministic)  │
  │     min tools / preferred / chart / err │
  │                                         │
  │  2. Pack judge context                  │
  │     • this-turn tools → context relev.  │
  │     • session + tools (+ prior turns    │
  │       in Streamlit) → groundedness / AR │
  │                                         │
  │  3. LLM-as-judge (temp=0, JSON)         │
  │     ┌───────────────────────────────┐   │
  │     │ Answer relevance     ──► 0–1  │   │
  │     │ Groundedness         ──► 0–1  │──►│ rag_triad_mean
  │     │ Context relevance    ──► 0–1  │   │   = mean of three
  │     └───────────────────────────────┘   │
  │     Execution efficiency   ──► 0–1      │  (GPA-inspired;
  │                                         │   not in triad mean)
  └──────────────────┬──────────────────────┘
                     │
                     ▼
        Streamlit sidebar  /  evals/results/*.json
```

**RAG Triad** catches answer quality failures (off-topic / invented / bad
retrieval). **Execution efficiency** scores process quality (lean tool use).
`--no-judge` skips step 3 and keeps only expectation checks.

## Notes

- **Concurrent users:** each browser login is an isolated Streamlit session
  (own chat, active schema, MCP subprocess). Logging out closes **only that**
  session's Snowflake connection — other users keep working. All sessions still
  share the same MCP service account from `.env` (warehouse quotas apply).
- The agent is restricted to **read-only** exploration; it issues `SELECT`
  queries only and adds `LIMIT` clauses to avoid pulling huge result sets.
- **Charts**: ask for a chart/graph/pie/plot and the agent queries the data,
  then renders it inline (bar, line, area, scatter, or pie) via Altair.
- Keep your `.env` and private keys out of version control.
