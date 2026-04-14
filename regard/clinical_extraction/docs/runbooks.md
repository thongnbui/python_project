# Runbooks (prototype)

## Embedding provider errors (5xx / rate limit)

1. Confirm status page and API key validity.
2. Switch eval to `--dry-run` to unblock CI; tag release as “eval only.”
3. Reduce batch size and enable exponential backoff on ingestion jobs.
4. Page owner listed in `prompts/manifest.yaml`.

## Vector / lexical index degradation

1. Check index health dashboards (latency, error rate).
2. Fail closed on retrieve path: return `insufficient_context` with user-safe message.
3. Trigger index rebuild from last known good snapshot if corruption suspected.

## RAG index backup / restore (stub)

Contract fields live in [`rag/ingestion_contract.yaml`](../rag/ingestion_contract.yaml). For each vector and lexical store you adopt:

1. **Backup:** scheduled snapshot of index metadata + backing object store / shard files per tenant partition (see `vector_and_lexical.backup` in the YAML).
2. **Restore:** replay ingestion from last good `source_version` boundary, then run `evals/scripts/retrieval_metrics.py` and `evals/scripts/rag_ablation.py` on gold.
3. **Validate manifests:** `python evals/scripts/validate_ingestion_manifest.py path/to/manifest.jsonl` before bulk ingest.

## Model latency spike or 429 storms

1. Lower concurrency; enable request queue.
2. Temporary route to smaller model if rubric allows (update `model_allowlist` + re-run eval).
3. Enable cached retrieval embeddings and repeated-query cache where safe.

## Feature flags

- **Disable agent path:** fall back to single-shot `retrieve_chart` + one extraction call (document in deploy config).
- **Disable RAG:** pass empty `retrieved_chunks` only when policy allows chart-only mode.
- **YAML + env:** defaults in [`config/feature_flags.yaml`](../config/feature_flags.yaml); set `REGARD_FF_DISABLE_AGENT_PATH` / `REGARD_FF_FALLBACK_SINGLE_SHOT` to `1` in deploy; `run_eval.py` prints active flags to stderr.

## PHI + retention (reference)

- [`compliance/phi_minimization.yaml`](../compliance/phi_minimization.yaml) — hashing vs restricted debug store.
- [`compliance/retention_policy.yaml`](../compliance/retention_policy.yaml) — suggested retention windows and export-on-request.

## Failure review (pilot / prod)

- Cluster errors: `python evals/scripts/failure_review_summarize.py evals/runs/<id>/predictions.jsonl -o /tmp/review.md`

## Workflow tracing (OpenTelemetry)

- **Stub:** `agents/workflow_stub.py` emits `{otel_span_prefix}.workflow.run` and `{otel_span_prefix}.step.<tool_name>` (prefix from `agents/workflow.yaml`). Attributes use **`trace.id_hash`** (SHA-256 of the fixture `trace_id` string) and tool metadata only—no chart text or queries.
- **Production:** attach an OTLP or console exporter to the process `TracerProvider`; keep `phi_in_spans: false` in config and review span attributes in staging.

## Agent metrics (replay) and observability

- **Metrics (§4.3):** `run_fixture` returns `metrics` (trajectory length, duplicate-call stats, optional `expected_final` match, tool-error counts, recovery flags). **Batch / p95:**  
  `python agents/workflow_stub.py --feature-root . --fixtures-dir agents/fixtures` (from `clinical_extraction/`).
- **Structured logs (§4.4):** `REGARD_AGENT_STRUCTURED_LOG=1` and optional `REGARD_AGENT_LOG_PATH=/path/to.ndjson` — one JSON object per tool step (`trace_id_hash`, `duration_ms`, `outcome`, `retry_count`). No chart text.
- **Debug bundle:** `python agents/export_debug_bundle.py agents/fixtures/trace_ce-003.json -o /tmp/bundle.json` (internal-only handoff; use `--redact` for safer sharing).
