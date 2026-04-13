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

## 4. Freeze a baseline (regression gate)

After a good run:

```bash
cp regard/clinical_extraction/evals/runs/<good_run_id>/metrics.json \
   regard/clinical_extraction/evals/baseline_metrics.json
```

Later:

```bash
python regard/clinical_extraction/evals/scripts/run_eval.py --dry-run --include-stress \
  --baseline-metrics regard/clinical_extraction/evals/baseline_metrics.json
```

## 5. RAG retrieval metrics (offline)

```bash
python regard/clinical_extraction/evals/scripts/retrieval_metrics.py
```

Uses gold rows that define `gold_chunk_ids` (e.g. `ce-003`).
