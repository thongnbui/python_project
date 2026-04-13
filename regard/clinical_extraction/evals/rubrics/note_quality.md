# Note quality and extraction rubric (human / LLM-judge)

Use this rubric for **spot checks** and **LLM-as-judge** prompts. Calibrate judges against human labels before trusting scores for gates.

## Scale (per case)

| Score | Meaning |
|-------|---------|
| 2 | All extracted facts clearly supported by excerpt or cited chunk; no unsafe additions. |
| 1 | Minor omission or wording drift; no unsafe fabrication. |
| 0 | Unsupported fact, wrong patient context, or policy violation. |

## LLM-judge instructions (paste as system)

You grade clinical extraction JSON against the provided chart excerpt and optional chunks. You only judge **support** from the text, not medical correctness beyond obvious contradictions.

Output JSON: `{"score": 0|1|2, "rationale": "...", "unsupported_claims": ["..."]}`

Do not leak `notes_for_judge` into the rationale if they contain hidden answers; use only excerpt and model output.

## Human reviewer checklist

- [ ] Each entity has traceable support (quote or chunk).
- [ ] No new medications or diagnoses not in source.
- [ ] `insufficient_context` used appropriately when text is silent.
- [ ] Language is professional; no alarming statements beyond source.
