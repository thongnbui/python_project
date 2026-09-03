# Agent evals

Headless golden-query harness for the Snowflake MCP data agent. Inspired by the
DeepLearning.AI *Building and Evaluating Data Agents* course (RAG Triad +
process/efficiency scoring), adapted for OpenAI tool-calling instead of
LangGraph/TruLens.

Implementation lives in `metrics.py` (scoring), `runner.py` / `run_eval.py`
(headless runs), and `golden_queries.json` (cases). The Streamlit sidebar
**Last-turn eval** panel reuses the same `score_trace` helpers.

## Prerequisites

Same `.env` as the Streamlit app for the **MCP service account**:

- `OPENAI_API_KEY`
- `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_PRIVATE_KEY_FILE`
- Optional: `SNOWFLAKE_SCHEMA` (required for queries tagged with `requires_schema`)
- Optional: `EVAL_JUDGE_MODEL` (default `gpt-4o`)

The Streamlit login username/password are entered in the UI only and are not
used by the harness.

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

## Why these metrics?

A Snowflake data agent is not a plain chat bot: it **retrieves** (MCP tools /
SQL), then **generates** an answer. That maps cleanly to the course’s **RAG
Triad** — three complementary failure modes that a single “was the answer good?”
score would hide:

| Failure mode | What goes wrong | Metric that catches it |
|--------------|-----------------|------------------------|
| Fluent but off-topic | Sounds helpful, ignores the ask | **Answer relevance** |
| Fluent but invented | Names tables/numbers not in evidence | **Groundedness** |
| Bad retrieval / no tools | Wrong or empty tool results for the ask | **Context relevance** |

Those three are averaged into `rag_triad_mean`. Separately we keep:

- **Expectation checks** — cheap, deterministic gates for CI / smoke runs
  (`--no-judge`) without paying for LLM judges.
- **Execution efficiency** — course **GPA**-inspired process quality (lean tool
  use). Not part of the triad mean; it answers “did we get there wastefully?”
  even when the answer itself is fine.

### What GPA is

**GPA** = **Goal–Plan–Act**. It comes from the same DeepLearning.AI course and
evaluates whether the agent’s *process* hangs together — not only whether the
final answer looks good.

| Piece | Question | Failure mode |
|-------|----------|--------------|
| **Goal** | Did we stay aligned with what the user asked? | Drift, partial answer, wrong objective |
| **Plan** | Was there a clear, actionable approach? | Vague steps, missing constraints |
| **Act** | Did the tools/actions follow that plan and serve the goal? | Off-plan calls, redundant retries, irrelevant tools |

In the course, those dimensions are often scored separately (plan quality,
action relevance, goal alignment) with an LLM judge over a LangGraph-style
trace that includes an explicit plan step.

**How it maps here:** this agent has no separate planner — only an OpenAI tool
loop. We do **not** ship three GPA scores. Instead, **execution efficiency** is
the GPA-inspired stand-in: it judges the **tool trace** (the *Act* side) for a
lean, appropriate process given the user’s goal (necessary tools, little
redundancy, few errors; low when warehouse facts were needed but tools = 0).

In short:

- **RAG Triad** → quality of retrieve + answer (relevance, groundedness, context)
- **GPA / execution efficiency** → quality of *how* the agent got there

We use **OpenAI LLM-as-judge** (not TruLens) so the harness stays lightweight and
matches this repo’s OpenAI tool loop rather than LangGraph/TruGraph.

### What “context” means here

Tool outputs are packed into a text block for judges (`tool_contexts` /
`judge_context` in `metrics.py`):

- High-signal tools first (`read_query`, `describe_table`, charts, then inventory).
- `list_tables` / `describe_table` are **compacted** to name lists so large
  schemas still fit a ~16k character budget.
- Time-series / row payloads keep **head and tail** (not only the first rows),
  and `read_query` includes a truncated `sql=` so table names stay in evidence.
  Otherwise a monthly chart’s first 8 points can look like “only 2023” while
  the answer correctly describes 2024 in the tail.
- **Groundedness / answer relevance** may also see a `[session]` block (login
  default DB/schema, active schema) plus, in the Streamlit UI, **prior turns’
  tool steps** so multi-turn reuse is not always scored as invention.
- **Context relevance** and **efficiency** use **this turn’s tools only**. If
  this turn called no tools, context relevance is fixed at **0.0** (nothing newly
  retrieved).

---

## Metrics (detail)

| Metric | Type | In triad mean? |
|--------|------|----------------|
| Expectation checks | Deterministic | No (pass/fail) |
| Answer relevance | LLM judge → `[0, 1]` | Yes |
| Groundedness | LLM judge → `[0, 1]` | Yes |
| Context relevance | LLM judge → `[0, 1]` (or fixed 0) | Yes |
| Execution efficiency | LLM judge → `[0, 1]` | No |
| `rag_triad_mean` | Mean of the three RAG scores | — |

All LLM judges call `EVAL_JUDGE_MODEL` at `temperature=0` with
`response_format=json_object`, expecting
`{"score": <float>, "reason": "..."}`. Scores are clamped to `[0, 1]`.

### 1. Expectation checks (deterministic)

**Why:** Fast regression signal without judge cost or variance. Encode “this
golden query must at least list tables / draw a chart / not error.”

**How (`check_expectations`):** From the tool trace, compute:

- `tool_call_count`, `unique_tools`, `error_count`, `chart_count`

Then boolean checks against the case’s `expect` block in `golden_queries.json`:

| Check | Pass condition |
|-------|----------------|
| `min_tool_calls` | `tool_call_count >= expect.min_tool_calls` |
| `used_any_preferred_tool` | Intersection with `preferred_tools` (or skip if empty) |
| `require_chart` | At least one `display_chart` step if `require_chart: true` |
| `no_tool_errors` | No step has `error` |

`expectations.passed` is true only if **all** checks pass.

### 2. Answer relevance (RAG Triad)

**Why:** Separates “did we address the user?” from “did we invent facts?” A
hallucinated but on-topic summary still scores high here — groundedness must
catch the rest.

**Inputs:** question, answer, optional supporting context (session + tools).

**How:** Judge scores 0–1 how well the answer addresses the question. Prompt
notes for this agent: if the user scoped a `database.schema` and asked to pick
*a* table/column inside it, any real object from that schema is fine; do not
require three-part names unless the question did.

**High when:** The reply covers what was asked (inventory, summary, chart, DQ,
etc.).

**Low when:** Partial, evasive, or answering a different question.

### 3. Groundedness (RAG Triad)

**Why:** The main anti-hallucination metric for warehouse agents. Answer
relevance alone is insufficient — models write convincing schema summaries from
memory.

**Inputs:** question, answer, **allowed context** = `[session]` + compacted tool
outputs (and prior-turn tools in the Streamlit last-turn panel).

**How:** Judge scores 0–1 whether claims are supported by that context. Session
facts (login default DB/schema, active schema) **count as evidence**. Compacted
lines like `tables(N): db.schema.T1, ...` are treated as full inventories.
Invented tables, columns, numbers, or DBs not in context are penalized.

**High when:** Named objects and figures appear in tool/session evidence.

**Low when:** No tool context and the answer asserts specific warehouse facts
(classic “Tools: 0” failure).

### 4. Context relevance (RAG Triad)

**Why:** Isolates retrieval quality. The agent can answer well (AR) and stick to
evidence (G) only if the tools it called were on-topic — or fail G/CR when it
called nothing.

**Inputs:** question + **this turn’s** tool context only (no session block).

**How:**

1. If this turn has **no tool calls** → score **0.0** immediately (no LLM call),
   with reason that nothing was newly retrieved.
2. Otherwise an LLM judge scores 0–1 whether tool results are **on-topic and
   useful toward** the question — **not** whether they alone are complete enough
   to fully answer it.

Incomplete-but-relevant inventory (`list_tables` for a summary/overview ask)
should score **high**. Near-zero only for empty, failed, or clearly off-topic
tools.

**High when:** Tools match the ask (e.g. `list_tables` for “what’s in this
schema?”, `read_query` for counts).

**Low when:** Wrong schema, unrelated tools, or **zero tools this turn**.

### 5. Execution efficiency (GPA-inspired)

**Why:** See [What GPA is](#what-gpa-is). Goal–Plan–Act stresses coherent
process, not only final answer quality. Two agents can both score a strong
triad; the one that retries the same failing SQL ten times is worse in
production. This metric is our single GPA stand-in (tool-trace / *Act* quality).

**Inputs:** question, compact tool-trace summary, deterministic efficiency
stats (`tool_call_count`, errors, charts, unique tools).

**How:** LLM judge scores 0–1 for a lean, appropriate process. Prompt guidance:

- High: necessary, non-redundant tools; few errors.
- Low: repeated failures, pointless calls, or **zero tools when the question
  needs live Snowflake facts** (list/find tables, summarize data, propose a
  star schema from real tables) — target ≤ 0.4.
- Pure design asks with no warehouse object claims may still score high with
  zero tools.

**Not** included in `rag_triad_mean`.

### 6. `rag_triad_mean`

```text
rag_triad_mean = mean(answer_relevance, groundedness, context_relevance)
```

Use it as a single regression number when comparing `--app-version` labels.
Always inspect the three components when it drops — e.g. **0.33** with AR=1 /
G=0 / CR=0 usually means “looks good, no tools this turn.”

---

## Interpreting last-turn scores (Streamlit)

| Sidebar signal | Likely meaning |
|----------------|----------------|
| Tools = 0, CR = 0, G = 0 | Answered from memory; triad mean ≈ AR/3 |
| Tools = 0, CR = 0, G high | Prior-turn tools / session supported claims; CR still 0 by design |
| AR high, G/CR high | Healthy retrieve-then-answer turn |
| AR high, G low, tools > 0 | Answer went beyond what tools returned |

Clear conversation after prompt changes so the agent picks up updated system
instructions (e.g. re-call tools instead of relying on chat memory).

---

## Extending

Edit `golden_queries.json` to add cases for your schemas. Each case can set
`expect` (deterministic gates), `tags`, and `requires_schema`.
