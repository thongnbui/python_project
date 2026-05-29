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

   This creates a project-local `.venv/` folder containing its own `streamlit`,
   `openai`, `mcp`, etc. (Note: a `.venv` is tied to its absolute path — if you
   move or rename the folder, recreate it with the commands above.)

3. **Configure the OpenAI key.** Copy the example env file and fill it in:

```bash
cp .env.example .env
```

The only **required** variable is `OPENAI_API_KEY` (the app's LLM key).

**Snowflake credentials are entered on the app's login page**, not in `.env`.
Any `SNOWFLAKE_*` values you do set in `.env` are used only to *pre-fill* the
login form for convenience (handy for non-secret fields like
`SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`,
`SNOWFLAKE_ROLE`). Leave `SNOWFLAKE_USER`/`SNOWFLAKE_PASSWORD` out to avoid
storing secrets on disk.

On the login screen you provide: **Account**, **User**, **Password**,
**Warehouse**, **Database** (required), plus optional **Role** and **Schema**
(leave Schema blank to explore **all** schemas — it defaults to
`INFORMATION_SCHEMA` just to satisfy the connection). Use **Log out** in the
sidebar to switch accounts.

4. **Run the app** using the venv's Streamlit directly:

```bash
.venv/bin/streamlit run app.py
```

   Or activate the venv first and run it by name:

```bash
source .venv/bin/activate
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
