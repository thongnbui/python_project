#!/usr/bin/env python3
"""LLM-as-judge (or offline proxy) for extraction quality per evals/rubrics/note_quality.md.

* **Live:** blinded prompt (chart + chunks + model JSON only; no gold slots, no
  ``notes_for_judge``). Structured JSON 0–2 score + rationale.
* **Dry-run:** deterministic **proxy** from parse_ok, stress_ok, entity grounding
  (``claim_support_report``), and ``entity_recall`` — for CI without API keys.
* **Calibration:** optional ``--human-scores`` JSONL with ``case_id`` + ``score``;
  reports **match rate**, **Cohen's κ** (unweighted), and **linear weighted κ** (ordinal 0–2).
  Optional ``--min-calibration-n N`` exits non-zero when overlap count is below *N*
  (requires ``--human-scores``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from openai import OpenAI

from regard.clinical_extraction.schemas.models import ExtractionOutput, JudgeOutput

FEATURE_ROOT = Path(__file__).resolve().parents[2]
JUDGE_SCHEMA_PATH = FEATURE_ROOT / "schemas" / "judge_output.json"
OPENAI_JUDGE_SCHEMA_PATH = FEATURE_ROOT / "schemas" / "judge_output_openai.json"
DEFAULT_RUBRIC = FEATURE_ROOT / "evals" / "rubrics" / "note_quality.md"


def _load_env_files(feature_root: Path) -> None:
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    load_dotenv(feature_root / ".env", override=True)


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_gold_by_case_id(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        if not p.is_file():
            print(f"Warning: gold file not found: {p}", file=sys.stderr)
            continue
        for row in _load_jsonl(p):
            cid = row.get("case_id")
            if cid:
                out[str(cid)] = row
    return out


def _parse_predictions_path(arg: Path) -> Path:
    if arg.is_dir():
        p = arg / "predictions.jsonl"
        if not p.is_file():
            raise SystemExit(f"No predictions.jsonl under {arg}")
        return p
    if not arg.is_file():
        raise SystemExit(f"Not a file or run directory: {arg}")
    return arg


def _load_claim_support_module() -> Any:
    path = Path(__file__).resolve().parent / "claim_support_report.py"
    spec = importlib.util.spec_from_file_location("_claim_support_report", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _cohen_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Unweighted Cohen's kappa for paired integer labels (same length)."""
    if len(labels_a) != len(labels_b) or not labels_a:
        return float("nan")
    n = len(labels_a)
    agree = sum(1 for x, y in zip(labels_a, labels_b) if x == y)
    p_o = agree / n
    cats = sorted(set(labels_a) | set(labels_b))

    def p_marg(xs: list[int], c: int) -> float:
        return sum(1 for x in xs if x == c) / n

    p_e = sum(p_marg(labels_a, c) * p_marg(labels_b, c) for c in cats)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _linear_weighted_kappa(
    labels_a: list[int],
    labels_b: list[int],
    *,
    min_score: int = 0,
    max_score: int = 2,
) -> float:
    """Linear weighted Cohen's kappa for ordinal scores on a fixed scale.

    Weight ``w(i,j) = 1 - |i-j| / (max_score - min_score)``; chance uses both
    raters' marginals (same form as unweighted κ but with weighted agreement).
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return float("nan")
    n = len(labels_a)
    scale = max_score - min_score
    if scale <= 0:
        return float("nan")

    def w(i: int, j: int) -> float:
        return 1.0 - abs(i - j) / scale

    p_o = sum(w(a, b) for a, b in zip(labels_a, labels_b)) / n

    ca = Counter(labels_a)
    cb = Counter(labels_b)
    cats = sorted(set(labels_a) | set(labels_b))
    p_e = 0.0
    for i in cats:
        for j in cats:
            p_e += w(i, j) * (ca[i] / n) * (cb[j] / n)

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(1.0 - p_o) < 1e-12 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _validate_judge_schema(instance: dict[str, Any]) -> None:
    schema = json.loads(JUDGE_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(instance)


def _strip_for_judge(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "entities": body.get("entities", []),
        "citations": body.get("citations", []),
        "insufficient_context": body.get("insufficient_context", False),
        "clarifying_questions": body.get("clarifying_questions", []),
    }


def _dry_proxy_score(
    *,
    case: dict[str, Any],
    pred_row: dict[str, Any],
    pred: ExtractionOutput | None,
    csr: Any,
) -> tuple[int, str, str]:
    """Return (score, method, rationale) for offline proxy."""
    if not case:
        return 0, "proxy_missing_gold", "no gold row; cannot score grounding proxy"
    if not pred_row.get("parse_ok"):
        return 0, "proxy_parse", "parse_ok=false"
    if pred_row.get("stress_ok") is False:
        return 0, "proxy_stress", "stress_ok=false"
    if pred is None:
        return 0, "proxy_parse", "could not parse model output"

    corpus = csr.build_support_corpus(case, pred)
    unsupported: list[str] = []
    for ent in pred.entities:
        val = str(ent.get("value", "") or "")
        if not val:
            continue
        if not csr.claim_supported(val, corpus, token_fallback=True):
            unsupported.append(val)
    if unsupported:
        return (
            0,
            "proxy_grounding",
            f"unsupported entity values vs excerpt/chunks: {unsupported[:5]}",
        )

    rec = pred_row.get("entity_recall")
    if rec is not None and float(rec) < 0.999:
        return 1, "proxy_recall", f"entity_recall={rec} (minor omission vs gold)"

    return 2, "proxy_full", "parse_ok, grounding ok, recall full (or N/A)"


def _call_judge_openai(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    use_strict_json_schema: bool,
) -> tuple[JudgeOutput, int | None, int | None, str]:
    client = OpenAI()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    common: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": 512,
        "messages": messages,
    }

    if use_strict_json_schema and OPENAI_JUDGE_SCHEMA_PATH.is_file():
        schema_obj = json.loads(OPENAI_JUDGE_SCHEMA_PATH.read_text(encoding="utf-8"))
        try:
            resp = client.chat.completions.create(
                **common,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_output",
                        "strict": True,
                        "schema": schema_obj,
                    },
                },
            )
            content = resp.choices[0].message.content or "{}"
            usage = resp.usage
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            body = json.loads(content)
            _validate_judge_schema(body)
            return JudgeOutput.model_validate(body), pt, ct, "json_schema"
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: judge json_schema failed ({exc}); falling back to json_object.",
                file=sys.stderr,
            )

    resp = client.chat.completions.create(
        **common,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    usage = resp.usage
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    ct = getattr(usage, "completion_tokens", None) if usage else None
    body = json.loads(content)
    _validate_judge_schema(body)
    return JudgeOutput.model_validate(body), pt, ct, "json_object"


def _build_user_payload(case: dict[str, Any], body: dict[str, Any]) -> str:
    chart = str((case.get("input") or {}).get("chart_excerpt", "") or "")
    chunks = case.get("retrieved_chunks") or []
    chunk_lines = []
    for i, ch in enumerate(chunks):
        if isinstance(ch, dict):
            cid = ch.get("chunk_id", f"chunk_{i}")
            txt = ch.get("text", "")
            chunk_lines.append(f"- {cid}: {txt}")
    payload = {
        "case_id": case.get("case_id"),
        "chart_excerpt": chart,
        "retrieved_chunks": chunk_lines,
        "model_output": _strip_for_judge(body),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _load_human_scores(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in _load_jsonl(path):
        cid = str(row.get("case_id", ""))
        if not cid:
            continue
        s = row.get("score")
        if s is None:
            continue
        si = int(s)
        if si not in (0, 1, 2):
            raise SystemExit(f"Invalid human score for {cid}: {s} (want 0–2)")
        out[cid] = si
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions_jsonl_or_run_dir",
        type=Path,
        help="predictions.jsonl or run directory",
    )
    parser.add_argument(
        "--gold-jsonl",
        action="append",
        default=[],
        metavar="PATH",
        help="Gold JSONL (repeat). Default: main + stress gold.",
    )
    parser.add_argument(
        "--human-scores",
        type=Path,
        default=None,
        help="JSONL with case_id + score (0–2) for calibration vs judge output.",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=DEFAULT_RUBRIC,
        help="Markdown rubric (injected into system prompt for live mode).",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="JSON report path")
    parser.add_argument("--dry-run", action="store_true", help="No API; proxy scores only")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--no-strict-schema",
        action="store_true",
        help="Skip json_schema strict (live only); use json_object.",
    )
    parser.add_argument(
        "--min-calibration-n",
        type=int,
        default=None,
        metavar="N",
        help="With --human-scores, exit 1 if overlapping case count < N.",
    )
    args = parser.parse_args()

    if args.min_calibration_n is not None and args.human_scores is None:
        print(
            "--min-calibration-n requires --human-scores",
            file=sys.stderr,
        )
        return 2
    if args.min_calibration_n is not None and args.min_calibration_n < 0:
        print("--min-calibration-n must be >= 0", file=sys.stderr)
        return 2

    pred_path = _parse_predictions_path(args.predictions_jsonl_or_run_dir)
    if not args.gold_jsonl:
        args.gold_jsonl = [
            FEATURE_ROOT / "evals" / "gold" / "cases_v2026-04-13.jsonl",
            FEATURE_ROOT / "evals" / "gold" / "stress.jsonl",
        ]
    gold = _load_gold_by_case_id([Path(p) for p in args.gold_jsonl])
    human_by_id = _load_human_scores(args.human_scores) if args.human_scores else {}

    csr = _load_claim_support_module()
    rubric_text = (
        args.rubric.read_text(encoding="utf-8") if args.rubric.is_file() else ""
    )
    system_live = (
        "You grade clinical extraction JSON. Use ONLY chart_excerpt, "
        "retrieved_chunks, and model_output below. Do not infer from hidden gold.\n\n"
        f"--- RUBRIC ---\n{rubric_text[:12000]}\n--- END RUBRIC ---"
    )

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY", "").strip():
        print(
            "Live judge needs OPENAI_API_KEY or use --dry-run.",
            file=sys.stderr,
        )
        return 2

    if not args.dry_run:
        _load_env_files(FEATURE_ROOT)

    per_case: list[dict[str, Any]] = []
    machine_by_id: dict[str, int] = {}
    n = 0
    for row in _load_jsonl(pred_path):
        if args.max_cases is not None and n >= args.max_cases:
            break
        n += 1
        cid = str(row.get("case_id", ""))
        case = gold.get(cid)
        entry: dict[str, Any] = {"case_id": cid, "mode": "dry_run" if args.dry_run else "live"}

        pred: ExtractionOutput | None = None
        if row.get("parse_ok"):
            try:
                body = json.loads(row.get("raw") or "{}")
                body.setdefault(
                    "_meta",
                    {"prompt_version": "unknown", "model": "unknown"},
                )
                pred = ExtractionOutput.model_validate(body)
            except Exception as exc:  # noqa: BLE001
                entry["parse_error"] = str(exc)

        if args.dry_run:
            score, method, why = _dry_proxy_score(
                case=case if case is not None else {},
                pred_row=row,
                pred=pred,
                csr=csr,
            )
            entry["score"] = score
            entry["method"] = method
            entry["rationale"] = why
            machine_by_id[cid] = score
        else:
            if case is None:
                entry["score"] = 0
                entry["method"] = "skipped_missing_gold"
                entry["rationale"] = "no gold row for case_id"
                machine_by_id[cid] = 0
            elif not row.get("parse_ok") or pred is None:
                score, method, why = _dry_proxy_score(
                    case=case,
                    pred_row=row,
                    pred=pred,
                    csr=csr,
                )
                entry["score"] = score
                entry["method"] = method
                entry["rationale"] = why
                machine_by_id[cid] = score
            else:
                body = json.loads(row.get("raw") or "{}")
                user = _build_user_payload(case, body)
                try:
                    jo, pt, ct, fmt = _call_judge_openai(
                        system=system_live,
                        user=user,
                        model=args.model,
                        temperature=args.temperature,
                        use_strict_json_schema=not args.no_strict_schema,
                    )
                    entry["score"] = jo.score
                    entry["method"] = f"openai_{fmt}"
                    entry["rationale"] = jo.rationale
                    entry["unsupported_claims"] = jo.unsupported_claims
                    entry["tokens"] = {"prompt": pt, "completion": ct}
                    machine_by_id[cid] = jo.score
                except Exception as exc:  # noqa: BLE001
                    entry["score"] = 0
                    entry["method"] = "openai_error"
                    entry["rationale"] = str(exc)
                    machine_by_id[cid] = 0

        per_case.append(entry)

    # Calibration vs human
    calibration: dict[str, Any] = {"human_scores_file": None, "n_overlap": 0}
    if human_by_id:
        calibration["human_scores_file"] = str(args.human_scores)
        paired_a: list[int] = []
        paired_b: list[int] = []
        for cid, hs in human_by_id.items():
            if cid not in machine_by_id:
                continue
            paired_a.append(hs)
            paired_b.append(machine_by_id[cid])
        calibration["n_overlap"] = len(paired_a)
        if paired_a:
            matches = sum(1 for a, b in zip(paired_a, paired_b) if a == b)
            calibration["match_rate"] = round(matches / len(paired_a), 4)
            calibration["cohens_kappa"] = round(float(_cohen_kappa(paired_a, paired_b)), 4)
            calibration["cohens_kappa_linear_weighted"] = round(
                float(_linear_weighted_kappa(paired_a, paired_b)),
                4,
            )

    mean_score = (
        sum(e["score"] for e in per_case) / len(per_case) if per_case else None
    )
    gate_failed = False
    if args.min_calibration_n is not None:
        no = int(calibration.get("n_overlap", 0))
        if no < args.min_calibration_n:
            print(
                f"Calibration overlap {no} < required --min-calibration-n "
                f"{args.min_calibration_n}",
                file=sys.stderr,
            )
            gate_failed = True

    report: dict[str, Any] = {
        "predictions": str(pred_path),
        "dry_run": args.dry_run,
        "model": args.model if not args.dry_run else None,
        "per_case": per_case,
        "aggregate": {
            "n_cases": len(per_case),
            "mean_score": round(mean_score, 4) if mean_score is not None else None,
        },
        "calibration": calibration,
    }
    if gate_failed:
        report["calibration_gate_failed"] = True
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 1 if gate_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
