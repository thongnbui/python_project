# Agent evals

Headless golden-query harness for the Snowflake MCP data agent. Inspired by the
DeepLearning.AI *Building and Evaluating Data Agents* course (RAG Triad +
process/efficiency scoring), adapted for OpenAI tool-calling instead of
LangGraph/TruLens.

## Prerequisites

Same `.env` as the Streamlit app, including Snowflake key-pair fields used for
login prefill:

- `OPENAI_API_KEY`
- `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_PRIVATE_KEY_FILE`
- Optional: `SNOWFLAKE_SCHEMA` (required for queries tagged with `requires_schema`)
- Optional: `EVAL_JUDGE_MODEL` (default `gpt-4o`)

## Run

From `snowflake_data_agent/`:

```bash
# Cheap smoke run (expectation checks only)
.venv/bin/python -m evals.run_eval --tags smoke --no-judge

# Full suite with LLM judges
.venv/bin/python -m evals.run_eval --app-version "v1: base"

# Single query
.venv/bin/python -m evals.run_eval --ids list_databases
```

Reports land in `evals/results/` (gitignored). Compare `app_version` labels
across runs when you change prompts or models.

## Metrics

| Metric | Type | Notes |
|--------|------|--------|
| Expectation checks | Deterministic | min tools, preferred tools, chart required, no errors |
| Answer relevance | LLM judge | RAG Triad |
| Groundedness | LLM judge | Answer vs tool/SQL **+ session** context (login default DB/schema allowed) |
| Context relevance | LLM judge | Tool results vs question (high-signal tools preferred in context budget) |
| Execution efficiency | LLM judge | GPA-inspired process quality |
| `rag_triad_mean` | Aggregate | Mean of the three RAG scores |

Edit `golden_queries.json` to add cases for your schemas.
