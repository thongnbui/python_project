#!/usr/bin/env python3
"""Batch evaluation for clinical_extraction: render prompts, call LLM or dry-run, validate, write runs/."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterator

# Repo root (…/python_project)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jsonschema import Draft202012Validator
from openai import OpenAI

from regard.clinical_extraction.schemas.models import ExtractionOutput

FEATURE_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = FEATURE_ROOT / "prompts"
SCHEMA_PATH = FEATURE_ROOT / "schemas" / "extraction_output.json"


def _load_env_files(feature_root: Path) -> None:
    """Load .env files; feature-local .env wins over repo root and prior shell exports."""
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
    load_dotenv(feature_root / ".env", override=True)


def _emit_active_feature_flags(feature_root: Path) -> None:
    """Log active deploy flags from ``config/feature_flags.yaml`` + env (no secrets)."""
    path = feature_root / "config" / "feature_flags.yaml"
    if not path.is_file():
        return
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    active: dict[str, bool] = {}
    for name, spec in (data.get("flags") or {}).items():
        envk = str(spec.get("env") or "")
        if envk and os.environ.get(envk, "").strip().lower() in ("1", "true", "yes", "on"):
            active[name] = True
    if active:
        print(
            json.dumps({"regard_feature_flags": active}, ensure_ascii=False),
            file=sys.stderr,
        )


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("GIT_SHA", "unknown")


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sha256_text(parts: list[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _load_prompt_bundle(prompt_entry: dict[str, Any]) -> tuple[str, Environment, Path]:
    sub = Path(prompt_entry["path"])
    base = PROMPTS_DIR / sub
    system = (base / "system.md").read_text(encoding="utf-8")
    env = Environment(
        loader=FileSystemLoader(str(base)),
        autoescape=select_autoescape(enabled_extensions=(".jinja",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return system, env, base


def _render_user(env: Environment, case: dict[str, Any]) -> str:
    tmpl = env.get_template("user.jinja")
    chart = case.get("input", {}).get("chart_excerpt", "")
    chunks = case.get("retrieved_chunks") or []
    return tmpl.render(chart_excerpt=chart, retrieved_chunks=chunks)


def _merge_meta(body: dict[str, Any], prompt_version: str, model: str) -> None:
    meta = body.setdefault("_meta", {})
    meta.setdefault("prompt_version", prompt_version)
    meta.setdefault("model", model)


def _dry_run_body(case: dict[str, Any], prompt_version: str, model: str) -> dict[str, Any]:
    cid = case["case_id"]
    exp = case.get("expected") or {}
    clarifying: list[str] = []
    citations: list[dict[str, str]] = []

    if cid == "stress-001":
        body = {
            "entities": [],
            "citations": [],
            "insufficient_context": True,
            "clarifying_questions": ["Provide a non-empty chart excerpt with clinical facts."],
        }
    elif cid == "stress-002":
        body = {
            "entities": [],
            "citations": [],
            "insufficient_context": True,
            "clarifying_questions": ["Instruction injection detected; no extraction performed."],
            "_meta": {"warnings": ["possible_prompt_injection"]},
        }
    elif cid == "stress-003":
        body = {
            "entities": [],
            "citations": [],
            "insufficient_context": False,
            "clarifying_questions": [],
        }
    else:
        entities = list(exp.get("entities", []))
        insufficient = bool(exp.get("insufficient_context", False))
        gold_chunk_ids = list(exp.get("gold_chunk_ids", []))
        if gold_chunk_ids:
            for gid in gold_chunk_ids:
                citations.append(
                    {"chunk_id": gid, "quote": "synthetic dry-run citation"}
                )
        elif entities:
            citations.append(
                {"chunk_id": "excerpt", "quote": "synthetic dry-run from excerpt"}
            )
        body = {
            "entities": entities,
            "citations": citations,
            "insufficient_context": insufficient,
            "clarifying_questions": clarifying,
        }

    if isinstance(body.get("_meta"), dict):
        inner = body["_meta"]
        inner.setdefault("prompt_version", prompt_version)
        inner.setdefault("model", model)
    else:
        _merge_meta(body, prompt_version, model)
    return body


def _call_openai(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    openai_schema_path: Path,
    use_strict_json_schema: bool,
) -> tuple[str, int | None, int | None, str]:
    """Call Chat Completions; prefer structured `json_schema` (no `_meta`), then full validate.

    Returns:
        content, prompt_tokens, completion_tokens, format_label (json_schema | json_object).
    """
    client = OpenAI()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    common = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if use_strict_json_schema and openai_schema_path.is_file():
        schema_obj = json.loads(openai_schema_path.read_text(encoding="utf-8"))
        try:
            resp = client.chat.completions.create(
                **common,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clinical_extraction_output",
                        "strict": True,
                        "schema": schema_obj,
                    },
                },
            )
            content = resp.choices[0].message.content or "{}"
            usage = resp.usage
            pt = getattr(usage, "prompt_tokens", None) if usage else None
            ct = getattr(usage, "completion_tokens", None) if usage else None
            return content, pt, ct, "json_schema"
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: structured json_schema call failed ({exc}); "
                "falling back to json_object.",
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
    return content, pt, ct, "json_object"


def _jsonschema_validate(instance: dict[str, Any]) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(instance)


def _normalize_entity_text(value: str) -> str:
    """Lowercase, collapse whitespace, trim trailing punctuation for recall."""
    s = str(value).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".,;:")


def _entity_recall(
    pred: ExtractionOutput,
    gold: list[dict[str, Any]],
    *,
    strict_text: bool,
) -> float:
    if not gold:
        return 1.0
    hits = 0
    for g in gold:
        want_t = str(g.get("type", "")).lower().strip()
        raw_v = str(g.get("value", ""))
        want_v = raw_v.lower().strip() if strict_text else _normalize_entity_text(raw_v)
        for e in pred.entities:
            pred_t = str(e.get("type", "")).lower().strip()
            pv = str(e.get("value", ""))
            got_v = pv.lower().strip() if strict_text else _normalize_entity_text(pv)
            if pred_t == want_t and got_v == want_v:
                hits += 1
                break
    return hits / len(gold)


def _stress_pass(case: dict[str, Any], pred: ExtractionOutput) -> bool:
    cid = case["case_id"]
    exp = case.get("expected") or {}
    if cid == "stress-001":
        return pred.insufficient_context and not pred.entities
    if cid == "stress-002":
        meds = [
            e
            for e in pred.entities
            if str(e.get("type", "")).lower() == "medication"
        ]
        joined = " ".join(str(e.get("value", "")).lower() for e in pred.entities)
        banned = str(exp.get("must_not_contain_medication", "")).lower()
        if banned and banned in joined:
            return False
        return not meds
    if cid == "stress-003":
        return len(pred.entities) == 0
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=FEATURE_ROOT,
        help="clinical_extraction root",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=FEATURE_ROOT / "evals" / "gold" / "cases_v2026-04-13.jsonl",
    )
    parser.add_argument(
        "--stress-dataset",
        type=Path,
        default=FEATURE_ROOT / "evals" / "gold" / "stress.jsonl",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="Also run stress.jsonl cases",
    )
    parser.add_argument("--prompt-id", default="clinical_extraction_v1")
    parser.add_argument("--run-id", default=None, help="Defaults to UTC-ish unique id")
    parser.add_argument("--dry-run", action="store_true", help="No API calls; synthetic outputs")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--force", action="store_true", help="Overwrite existing run directory")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
        help="Prior metrics.json; fail if primary metrics regress beyond delta",
    )
    parser.add_argument(
        "--regression-delta",
        type=float,
        default=0.02,
        help="Max allowed drop for schema_pass_rate and mean_entity_recall",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Only first N cases after merge (cheap live smoke test)",
    )
    parser.add_argument(
        "--no-strict-schema",
        action="store_true",
        help="Live only: skip OpenAI json_schema structured output; use json_object only",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        metavar="ID,ID,...",
        help="Only run these case_id values (order preserved). Implies you loaded them "
        "(include stress rows only if listed and --include-stress added them to the pool).",
    )
    parser.add_argument(
        "--strict-recall",
        action="store_true",
        help="Entity recall: exact string match (legacy). Default: normalized whitespace/case.",
    )
    args = parser.parse_args()

    feature_root = args.feature_root.resolve()
    _load_env_files(feature_root)
    _emit_active_feature_flags(feature_root)

    if not args.dry_run and not os.environ.get("OPENAI_API_KEY", "").strip():
        print(
            "Live eval needs OPENAI_API_KEY (env or .env). "
            "Use --dry-run for offline CI, or copy .env.example → .env.",
            file=sys.stderr,
        )
        return 2

    run_id = args.run_id or uuid.uuid4().hex[:12]
    runs_root = feature_root / "evals" / "runs" / run_id
    if runs_root.exists() and not args.force:
        print(f"Run dir exists: {runs_root} (use --force)", file=sys.stderr)
        return 2
    runs_root.mkdir(parents=True, exist_ok=True)

    manifest_path = feature_root / "prompts" / "manifest.yaml"
    manifest_raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    prompts_list = manifest_raw.get("prompts", [])
    prompt_entry = next(
        (p for p in prompts_list if p.get("prompt_id") == args.prompt_id), None
    )
    if not prompt_entry:
        print(f"Unknown prompt_id {args.prompt_id}", file=sys.stderr)
        return 1

    system, env, _ = _load_prompt_bundle(prompt_entry)
    prompt_version = f"{args.prompt_id}:{prompt_entry.get('version', '?')}"

    cases = list(_load_jsonl(args.dataset))
    if args.include_stress:
        cases.extend(list(_load_jsonl(args.stress_dataset)))
    if args.case_ids:
        want = [x.strip() for x in args.case_ids.split(",") if x.strip()]
        by_id = {c["case_id"]: c for c in cases}
        missing = [i for i in want if i not in by_id]
        if missing:
            print(
                f"Warning: case_id not found (skipped): {missing}",
                file=sys.stderr,
            )
        cases = [by_id[i] for i in want if i in by_id]
    if args.max_cases is not None:
        cases = cases[: max(0, args.max_cases)]

    allowlist = prompt_entry.get("model_allowlist") or []
    if (
        not args.dry_run
        and allowlist
        and args.model not in allowlist
    ):
        print(
            f"Warning: model {args.model!r} not in manifest model_allowlist {allowlist}.",
            file=sys.stderr,
        )

    predictions_path = runs_root / "predictions.jsonl"
    latencies: list[float] = []
    token_prompt_total = 0
    token_completion_total = 0
    schema_ok = 0
    recalls: list[float] = []
    stress_results: list[bool] = []
    strata_schema: dict[str, list[int]] = defaultdict(list)
    strata_recall: dict[str, list[float]] = defaultdict(list)

    openai_schema_path = feature_root / "schemas" / "extraction_output_openai.json"
    use_strict_openai_schema = not args.dry_run and not args.no_strict_schema
    openai_formats_used: list[str] = []

    with predictions_path.open("w", encoding="utf-8") as pred_f:
        for case in cases:
            case_id = case["case_id"]
            user = _render_user(env, case)
            sha = _sha256_text([system, user])
            t0 = time.perf_counter()
            raw = ""
            err = ""
            parse_ok = False
            pred: ExtractionOutput | None = None
            pt = ct = None

            try:
                if args.dry_run:
                    body = _dry_run_body(case, prompt_version, args.model)
                    _merge_meta(body, prompt_version, args.model)
                    raw = json.dumps(body)
                    _jsonschema_validate(body)
                    pred = ExtractionOutput.model_validate(body)
                else:
                    raw, pt, ct, fmt = _call_openai(
                        system=system,
                        user=user,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        openai_schema_path=openai_schema_path,
                        use_strict_json_schema=use_strict_openai_schema,
                    )
                    openai_formats_used.append(fmt)
                    body = json.loads(raw)
                    _merge_meta(body, prompt_version, args.model)
                    _jsonschema_validate(body)
                    pred = ExtractionOutput.model_validate(body)
                parse_ok = True
            except Exception as exc:  # noqa: BLE001 — collect per-case errors
                err = str(exc)
                raw = raw or json.dumps({"error": err})

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            if pt is not None:
                token_prompt_total += pt
            if ct is not None:
                token_completion_total += ct

            rec = None
            stress_ok = None
            if pred is not None:
                schema_ok += 1
                gold_entities = (case.get("expected") or {}).get("entities")
                if gold_entities is not None:
                    rec = _entity_recall(
                        pred,
                        gold_entities,
                        strict_text=args.strict_recall,
                    )
                    recalls.append(rec)
                if str(case_id).startswith("stress"):
                    stress_ok = _stress_pass(case, pred)
                    stress_results.append(stress_ok)
                for tag in case.get("rubric_tags") or []:
                    strata_schema[tag].append(1)
                    if rec is not None:
                        strata_recall[tag].append(rec)
            else:
                for tag in case.get("rubric_tags") or []:
                    strata_schema[tag].append(0)

            pred_f.write(
                json.dumps(
                    {
                        "case_id": case_id,
                        "prompt_sha256": sha,
                        "latency_ms": round(elapsed_ms, 3),
                        "parse_ok": parse_ok,
                        "error": err or None,
                        "raw": raw,
                        "entity_recall": rec,
                        "stress_ok": stress_ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(cases)
    schema_pass_rate = schema_ok / n if n else 0.0
    mean_recall = mean(recalls) if recalls else None
    stress_pass_rate = (
        mean(1.0 if x else 0.0 for x in stress_results) if stress_results else None
    )

    def _pct(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {"p50": 0.0, "p95": 0.0}
        s = sorted(xs)
        p50 = s[len(s) // 2]
        p95 = s[max(0, int(0.95 * (len(s) - 1)))]
        return {"p50": round(p50, 3), "p95": round(p95, 3)}

    metrics = {
        "run_id": run_id,
        "n_cases": n,
        "schema_pass_rate": round(schema_pass_rate, 4),
        "mean_entity_recall": round(mean_recall, 4) if mean_recall is not None else None,
        "stress_pass_rate": round(stress_pass_rate, 4)
        if stress_pass_rate is not None
        else None,
        "latency_ms": _pct(latencies),
        "tokens": {
            "prompt_total": token_prompt_total,
            "completion_total": token_completion_total,
        },
        "strata": {
            tag: {
                "schema_pass_rate": round(mean(strata_schema[tag]), 4)
                if strata_schema[tag]
                else None,
                "mean_entity_recall": round(mean(strata_recall[tag]), 4)
                if strata_recall[tag]
                else None,
            }
            for tag in sorted(set(strata_schema) | set(strata_recall))
        },
    }

    run_manifest = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "prompt_id": args.prompt_id,
        "prompt_version": prompt_entry.get("version"),
        "eval_dataset": str(args.dataset),
        "stress_dataset": str(args.stress_dataset),
        "include_stress": args.include_stress,
        "model": args.model,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        },
        "dry_run": args.dry_run,
        "max_cases": args.max_cases,
        "case_ids": [x.strip() for x in args.case_ids.split(",") if x.strip()]
        if args.case_ids
        else None,
        "entity_recall_mode": "strict" if args.strict_recall else "normalized",
        "openai_structured_output": {
            "strict_json_schema_requested": bool(use_strict_openai_schema),
            "schema_file": str(openai_schema_path)
            if use_strict_openai_schema
            else None,
            "formats_used": sorted(set(openai_formats_used))
            if openai_formats_used
            else None,
        },
    }
    (runs_root / "manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    (runs_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    if args.baseline_metrics and args.baseline_metrics.is_file():
        base = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
        delta = args.regression_delta
        if schema_pass_rate + delta < base.get("schema_pass_rate", 0):
            print("Regression: schema_pass_rate dropped", file=sys.stderr)
            return 1
        if mean_recall is not None and base.get("mean_entity_recall") is not None:
            if mean_recall + delta < base["mean_entity_recall"]:
                print("Regression: mean_entity_recall dropped", file=sys.stderr)
                return 1

    print(json.dumps(metrics, indent=2))

    if args.include_stress and stress_results and not all(stress_results):
        print("Stress checks failed for one or more cases.", file=sys.stderr)
        return 1

    return 0 if schema_ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
