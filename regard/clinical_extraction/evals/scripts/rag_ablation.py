#!/usr/bin/env python3
"""Offline RAG ablation: sweep ``top_k`` (precision@k) on gold rows with ``gold_chunk_ids``.

No live index — uses **chunk list order** already present in each dataset row. Writes JSON
(and optional CSV) under ``--output-dir`` for regression tracking (see playbook §5.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from regard.clinical_extraction.evals.scripts.retrieval_metrics import mean_precision_for_k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "gold"
        / "cases_v2026-04-13.jsonl",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="1,3,5,8",
        help="Comma-separated k values for precision@k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write metrics.json + sweep.csv under this directory",
    )
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.top_k_values.split(",") if x.strip()]
    rows_out: list[dict[str, Any]] = []
    for k in ks:
        n, avg = mean_precision_for_k(args.dataset, k)
        rows_out.append(
            {
                "top_k": k,
                "n_cases": n,
                f"mean_precision_at_{k}": round(avg, 6) if n else None,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "sweep": rows_out,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    sys.stdout.write(text)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plots = args.output_dir / "plots"
        plots.mkdir(exist_ok=True)
        (args.output_dir / "metrics.json").write_text(text, encoding="utf-8")
        csv_path = args.output_dir / "sweep.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["top_k", "n_cases", "mean_precision"])
            w.writeheader()
            for r in rows_out:
                k = r["top_k"]
                w.writerow(
                    {
                        "top_k": k,
                        "n_cases": r["n_cases"],
                        "mean_precision": r.get(f"mean_precision_at_{k}"),
                    }
                )
        placeholder = plots / "README.txt"
        placeholder.write_text(
            "Matplotlib plots optional: install matplotlib and plot sweep.csv here.\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output_dir / 'metrics.json'} and sweep.csv", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
