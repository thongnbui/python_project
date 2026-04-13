# Data flow (RAG + LLM) — review artifact

High-level flow for security and compliance reviews. Replace placeholders with your actual systems.

```mermaid
flowchart LR
  subgraph sources [Authorized sources]
    EHR[EHR export / FHIR]
  end
  subgraph pipeline [Ingestion]
    ING[De-ID / chunk / embed]
    IDX[(Vector + lexical index)]
  end
  subgraph runtime [Online]
    API[App / API]
    RET[Retrieve + filter]
    LLM[LLM vendor]
    LOG[Redacted logs + trace IDs]
  end
  EHR --> ING --> IDX
  API --> RET --> IDX
  RET --> LLM
  LLM --> API
  API --> LOG
```

## Notes

- **PHI minimization:** production logs should carry hashed patient/encounter IDs and omit raw note text except in restricted debug storage.
- **Deletion:** index rows must be tied to source document versions so corrections and deletes propagate.
- **Vendor:** document subprocessors and data processing agreements for the LLM provider used in `run_eval.py` live mode.
