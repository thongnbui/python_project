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
