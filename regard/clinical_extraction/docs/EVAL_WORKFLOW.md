# Eval workflow (recommended order)

## 1. Offline — dry run

Validates prompt rendering, JSON Schema, Pydantic, metrics, and stress checks **without** calling OpenAI.

```bash
cd /path/to/python_project
source regard/venv312/bin/activate
export PYTHONPATH="$PWD"
python regard/clinical_extraction/evals/scripts/run_eval.py --dry-run --include-stress
```

## 2. Live — smoke (one case)

Uses the real model; keeps cost low.

```bash
cp regard/clinical_extraction/.env.example regard/clinical_extraction/.env
# Edit .env and set OPENAI_API_KEY

# If you still see 429 or wrong-key behavior, a stale shell export may win; clear it:
# env -u OPENAI_API_KEY python regard/clinical_extraction/evals/scripts/run_eval.py ...

python regard/clinical_extraction/evals/scripts/run_eval.py --max-cases 1 --model gpt-4o-mini
```

`run_eval.py` loads repo-root `.env` first, then **`clinical_extraction/.env` with `override=True`**, so variables set in this folder’s `.env` (including `OPENAI_API_KEY`) replace the shell and repo root.

Inspect `regard/clinical_extraction/evals/runs/<run_id>/predictions.jsonl` for raw JSON and parse errors.

## 3. Live — full gold + stress

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py --include-stress --model gpt-4o-mini
```

### Subset of cases

Run only given `case_id`s (order preserved). Stress rows must be **included in the pool** (`--include-stress`) **and** listed if you want them:

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py --dry-run --include-stress \
  --case-ids ce-003,stress-001 --model gpt-4o-mini
```

### Export predictions for review

```bash
python regard/clinical_extraction/evals/scripts/export_predictions_csv.py \
  regard/clinical_extraction/evals/runs/<run_id>/predictions.jsonl -o /tmp/review.csv
```

With **gold labels** side-by-side (main + stress JSONL):

```bash
python regard/clinical_extraction/evals/scripts/export_predictions_csv.py \
  regard/clinical_extraction/evals/runs/<run_id>/predictions.jsonl \
  --gold-jsonl regard/clinical_extraction/evals/gold/cases_v2026-04-13.jsonl \
  --gold-jsonl regard/clinical_extraction/evals/gold/stress.jsonl \
  -o /tmp/review_with_gold.csv
```

Adds columns: `gold_chart_excerpt` (truncated), `gold_expected_json`, `gold_rubric_tags`, `gold_notes_for_judge`. Omit chart column with `--chart-excerpt-chars 0`.

### Claim support (grounding / hallucination triage)

[`evals/scripts/claim_support_report.py`](../evals/scripts/claim_support_report.py) labels each **entity value** as `supported` or `unsupported` against the merged evidence string: `input.chart_excerpt`, all `retrieved_chunks[].text`, and all `citations[].quote`. Default: normalized substring match, with an optional **token fallback** (significant tokens must each match the corpus, including a simple **y ↔ -ies** plural check). Use `--no-token-fallback` for stricter checks. If `predictions.jsonl` `raw` omits `_meta` (API text before harness merge), the script injects a placeholder so Pydantic parsing matches `run_eval.py` validation.

```bash
python regard/clinical_extraction/evals/scripts/claim_support_report.py \
  regard/clinical_extraction/evals/runs/<run_id>/predictions.jsonl \
  --gold-jsonl regard/clinical_extraction/evals/gold/cases_v2026-04-13.jsonl \
  --gold-jsonl regard/clinical_extraction/evals/gold/stress.jsonl \
  -o regard/clinical_extraction/evals/runs/<run_id>/claim_support.json
```

You can pass a **run directory** instead of `predictions.jsonl`. Optional gate: `--max-unsupported-rate 0.05` exits non-zero if the share of unsupported entity claims exceeds the threshold.

### LLM-as-judge (live) and dry proxy

[`evals/scripts/llm_judge_eval.py`](../evals/scripts/llm_judge_eval.py) scores each case on the **0–2 rubric** in [`evals/rubrics/note_quality.md`](../evals/rubrics/note_quality.md).

**Dry-run (CI, no API):** deterministic proxy from `parse_ok`, `stress_ok`, entity grounding (same heuristics as `claim_support_report.py`), and `entity_recall` when present.

```bash
python regard/clinical_extraction/evals/scripts/llm_judge_eval.py \
  regard/clinical_extraction/evals/runs/<run_id>/predictions.jsonl \
  --dry-run -o /tmp/judge_report.json
```

**Live judge:** requires `OPENAI_API_KEY` (same `.env` loading order as `run_eval.py`). Prompt is **blinded**: only `chart_excerpt`, `retrieved_chunks`, and parsed model output—no gold `expected` and no `notes_for_judge`.

```bash
python regard/clinical_extraction/evals/scripts/llm_judge_eval.py \
  regard/clinical_extraction/evals/runs/<run_id> \
  --model gpt-4o-mini \
  -o regard/clinical_extraction/evals/runs/<run_id>/judge_report.json
```

**Calibration:** add human labels (≥ N cases for a serious calibration story) as JSONL: one object per line with `case_id` and integer `score` in `{0,1,2}`.

```bash
python regard/clinical_extraction/evals/scripts/llm_judge_eval.py \
  regard/clinical_extraction/evals/runs/<run_id> \
  --dry-run \
  --human-scores regard/clinical_extraction/evals/human_scores.example.jsonl \
  -o /tmp/judge_calibrated.json
```

The report’s `calibration` block includes **`match_rate`** and **`cohens_kappa`** (unweighted) on cases present in both the run and `--human-scores`.

### Summarize & compare runs

```bash
python regard/clinical_extraction/evals/scripts/summarize_run.py \
  regard/clinical_extraction/evals/runs/<run_id>/predictions.jsonl \
  --metrics-json regard/clinical_extraction/evals/runs/<run_id>/metrics.json
```

```bash
python regard/clinical_extraction/evals/scripts/compare_metrics.py \
  regard/clinical_extraction/evals/baseline_metrics.json \
  regard/clinical_extraction/evals/runs/<run_id>/metrics.json
```

Use `--strict-exit` on `summarize_run.py` if you want a non-zero exit when any case failed parsing or stress checks.

## 4. Regression baseline (committed + optional live)

### Dry-run baseline (CI-friendly)

`evals/baseline_metrics.json` is checked in for **offline** regression. It records expected **`schema_pass_rate`** and **`mean_entity_recall`** after `run_eval.py --dry-run --include-stress`.

Check locally:

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py --dry-run --include-stress \
  --baseline-metrics regard/clinical_extraction/evals/baseline_metrics.json --force
```

`pytest regard/clinical_extraction/tests/test_baseline_regression.py` runs the same check.

**When to update the committed baseline:** you changed gold rows, stress cases, dry-run logic, or schema in a way that *intentionally* moves metrics. Regenerate:

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py --dry-run --include-stress --force
# Copy evals/runs/<new_id>/metrics.json → evals/baseline_metrics.json (merge _comment if desired)
```

### Live baseline (local only)

- [x] **Saved:** `evals/baseline_metrics_live.json` frozen from live run **`98bf4fe5c9ee`** (8 cases, `gpt-4o-mini`, `--include-stress`). *File is gitignored; recreate with the command below after future “golden” runs.*

After a good **live** run, keep a **private** copy for model/prompt A/B:

```bash
cp regard/clinical_extraction/evals/runs/<good_run_id>/metrics.json \
   regard/clinical_extraction/evals/baseline_metrics_live.json
```

`baseline_metrics_live.json` is **gitignored**. An older committed reference snapshot is in
[`evals/baseline_metrics_live.json.example`](../evals/baseline_metrics_live.json.example)
(run `aa7759fb70e8`, 6-case era).

Compare a new live run:

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py --include-stress --model gpt-4o-mini \
  --baseline-metrics regard/clinical_extraction/evals/baseline_metrics_live.json
```

## 5. Structured OpenAI output (default on live runs)

Live calls use **`json_schema` + `strict: true`** via
[`schemas/extraction_output_openai.json`](../schemas/extraction_output_openai.json)
(no `_meta` in the API schema; the harness injects `_meta` before full validation).

Disable if your model endpoint rejects structured outputs:

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py ... --no-strict-schema
```

`evals/runs/<id>/manifest.json` records `openai_structured_output.formats_used` (`json_schema` vs `json_object` fallback).

## 6. RAG retrieval metrics (offline)

```bash
python regard/clinical_extraction/evals/scripts/retrieval_metrics.py
```

Uses gold rows that define `gold_chunk_ids` (e.g. `ce-003`, `ce-005`).

## 7. CI (offline, no secrets)

GitHub Actions runs **`pytest regard/clinical_extraction/tests`** on changes under `regard/clinical_extraction/` (includes dry-run + committed `baseline_metrics.json` regression). No `OPENAI_API_KEY` required.

Workflow: [`.github/workflows/regard-clinical-extraction.yml`](../../../.github/workflows/regard-clinical-extraction.yml).
