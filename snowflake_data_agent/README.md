# ❄️ Snowflake Data Explorer (MCP chat agent)

A Streamlit chat app that connects to your Snowflake instance through the
[`mcp_snowflake_server`](https://github.com/isaacwasserman/mcp-snowflake-server)
Model Context Protocol (MCP) server. Ask questions in plain English and an
OpenAI model uses the MCP tools to explore your databases, schemas, tables and
their data.

## How it works

```
Streamlit UI  ──►  OpenAI (tool calling)  ──►  MCP client  ──►  Snowflake MCP server  ──►  Snowflake
   (app.py)            decides which tool        (mcp_client.py)     (mcp_snowflake_server)
```

The model is given the MCP server's tools (`list_databases`, `list_schemas`,
`list_tables`, `describe_table`, `read_query`, ...) and calls them in a loop
until it can answer. Tool calls are shown inline so you can see exactly what
SQL/lookups ran. Tabular results are rendered as dataframes.

`mcp_client.py` hosts the async MCP session on a background event-loop thread so
it can be driven from Streamlit's synchronous, rerun-on-every-click model.

## Setup

1. **Install `uv`** (used to launch the MCP server). See
   <https://docs.astral.sh/uv/>. On macOS: `brew install uv`.

2. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

3. **Configure credentials.** Copy the example env file and fill it in:

```bash
cp .env.example .env
```

Required: `OPENAI_API_KEY`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`,
`SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`.
Optional: `SNOWFLAKE_SCHEMA` (leave unset to explore **all** schemas in the
database — it defaults to `INFORMATION_SCHEMA` just to satisfy the connection),
`SNOWFLAKE_ROLE`.

4. **Run the app:**

```bash
streamlit run app.py
```

The first launch downloads the MCP server via `uvx`, which can take a minute.

## Try asking

- "List all databases I can access."
- "What tables are in the SNOWFLAKE_SAMPLE_DATA.TPCH_SF1 schema?"
- "Describe the CUSTOMER table and show me 10 sample rows."
- "Which 5 nations have the most customers?"

## Notes

- The agent is restricted to **read-only** exploration; it issues `SELECT`
  queries only and adds `LIMIT` clauses to avoid pulling huge result sets.
- Keep your `.env` out of version control (it contains secrets).
