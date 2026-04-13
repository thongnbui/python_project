# Gold eval sets

- **`cases_v2026-04-13.jsonl`** — canonical regression set for `clinical_extraction_v1`. Bump filename when you change rows (keep prior file archived).
- **`stress.jsonl`** — failure probes; use `run_eval.py --include-stress`.

Each line is one JSON object. Required fields are documented in the parent playbook [`../../README.md`](../../README.md) §3.4.
