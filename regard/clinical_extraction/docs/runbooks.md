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

## Model latency spike or 429 storms

1. Lower concurrency; enable request queue.
2. Temporary route to smaller model if rubric allows (update `model_allowlist` + re-run eval).
3. Enable cached retrieval embeddings and repeated-query cache where safe.

## Feature flags

- **Disable agent path:** fall back to single-shot `retrieve_chart` + one extraction call (document in deploy config).
- **Disable RAG:** pass empty `retrieved_chunks` only when policy allows chart-only mode.

## Workflow tracing (OpenTelemetry)

- **Stub:** `agents/workflow_stub.py` emits `{otel_span_prefix}.workflow.run` and `{otel_span_prefix}.step.<tool_name>` (prefix from `agents/workflow.yaml`). Attributes use **`trace.id_hash`** (SHA-256 of the fixture `trace_id` string) and tool metadata only—no chart text or queries.
- **Production:** attach an OTLP or console exporter to the process `TracerProvider`; keep `phi_in_spans: false` in config and review span attributes in staging.

## Agent metrics (replay) and observability

- **Metrics (§4.3):** `run_fixture` returns `metrics` (trajectory length, duplicate-call stats, optional `expected_final` match, tool-error counts, recovery flags). **Batch / p95:**  
  `python agents/workflow_stub.py --feature-root . --fixtures-dir agents/fixtures` (from `clinical_extraction/`).
- **Structured logs (§4.4):** `REGARD_AGENT_STRUCTURED_LOG=1` and optional `REGARD_AGENT_LOG_PATH=/path/to.ndjson` — one JSON object per tool step (`trace_id_hash`, `duration_ms`, `outcome`, `retry_count`). No chart text.
- **Debug bundle:** `python agents/export_debug_bundle.py agents/fixtures/trace_ce-003.json -o /tmp/bundle.json` (internal-only handoff; use `--redact` for safer sharing).
