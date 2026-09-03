# Snowflake Data Explorer (MCP chat agent)

A Streamlit chat app that connects to your Snowflake instance through the
[`mcp_snowflake_server`](https://github.com/isaacwasserman/mcp-snowflake-server)
Model Context Protocol (MCP) server. Ask questions in plain English and an
OpenAI model uses the MCP tools to explore your databases, schemas, tables and
their data.

## How it works

```
Streamlit UI  ──►  OpenAI (tool calling)  ──►  MCP client  ──►  Snowflake MCP server  ──►  Snowflake
   (app.py)            decides which tool        (mcp_client.py)     (mcp_snowflake_server)
                              ▲
                       agent_core.py
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

The Streamlit **login page** asks for a username and password (entered by the
user; not configured in `.env`). Warehouse access uses a separate **MCP service
account** from `.env` (key-pair / JWT):

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

## Notes

- The agent is restricted to **read-only** exploration; it issues `SELECT`
  queries only and adds `LIMIT` clauses to avoid pulling huge result sets.
- **Charts**: ask for a chart/graph/pie/plot and the agent queries the data,
  then renders it inline (bar, line, area, scatter, or pie) via Altair.
- Keep your `.env` and private keys out of version control.
