# Clinical extraction (sample vertical slice)

End-to-end scaffold for **grounded clinical concept extraction** from chart excerpts: prompts on disk, JSON Schema + Pydantic validation, gold eval sets, `run_eval.py`, RAG YAML, and a documented agent workflow.

| Field | Value |
|--------|--------|
| **Engineering owner** | _TBD_ |
| **Clinical reviewer** | _TBD_ |
| **Product link** | _TBD_ |
| **SLO (latency)** | p95 under _TBD_ ms per case (offline eval) |
| **SLO (cost)** | under _TBD_ USD per 1k cases at agreed model |
| **Rubric** | [`evals/rubrics/note_quality.md`](evals/rubrics/note_quality.md) |

## Quickstart

Use the shared **Python 3.12** environment under `regard/venv312` (created once at repo root):

```bash
cd /path/to/python_project
python3.12 -m venv regard/venv312   # if it does not exist yet
source regard/venv312/bin/activate   # Windows: regard\venv312\Scripts\activate
pip install -r regard/clinical_extraction/requirements.txt

cd regard/clinical_extraction
export PYTHONPATH="../.."   # repo root so `regard.*` imports work

# Dry run (no API key): validates harness + schema + metrics
python evals/scripts/run_eval.py --dry-run --include-stress

# Live smoke (1 case; loads OPENAI_API_KEY from clinical_extraction/.env or repo .env)
cp .env.example .env   # once; add your key
python evals/scripts/run_eval.py --max-cases 1 --model gpt-4o-mini

# Full live eval + stress
python evals/scripts/run_eval.py --include-stress --model gpt-4o-mini

# Tests (from repo root with venv active)
cd ../.. && pytest regard/clinical_extraction/tests -q
```

Step-by-step eval flow: [`docs/EVAL_WORKFLOW.md`](docs/EVAL_WORKFLOW.md).

## Layout

See the parent playbook [`../README.md`](../README.md) §2. This folder follows that layout.

## Related commands

| Script | Purpose |
|--------|---------|
| `evals/scripts/run_eval.py` | Batch eval → `evals/runs/<run_id>/` |
| `evals/scripts/retrieval_metrics.py` | Precision@k vs `gold_chunk_ids` on gold rows |
| `agents/workflow_stub.py` | Mocked vertical slice + fixture replay |
