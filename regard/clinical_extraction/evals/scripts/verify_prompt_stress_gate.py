#!/usr/bin/env python3
"""Gate: when ``manifest.yaml`` major.minor exceeds ``evals/gates/prompt_stress_baseline.txt``, exit 1.

After bumping **minor** (second component) or **major**, re-run stress evals, then update the
baseline file to match the new ``major.minor`` so CI documents intentional promotion.
Patch-only bumps (``1.0.1`` → ``1.0.2``) do not require a baseline update.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

FEATURE_ROOT = Path(__file__).resolve().parents[2]


def _major_minor_tuple(version: str) -> tuple[int, int]:
    parts = version.strip().split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 0
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=FEATURE_ROOT,
    )
    parser.add_argument("--prompt-id", default="clinical_extraction_v1")
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=None,
        help="Default: <feature-root>/evals/gates/prompt_stress_baseline.txt",
    )
    args = parser.parse_args()
    root = args.feature_root.resolve()
    manifest_path = root / "prompts" / "manifest.yaml"
    baseline_path = args.baseline_file or (root / "evals" / "gates" / "prompt_stress_baseline.txt")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    prompts = manifest.get("prompts") or []
    entry = next((p for p in prompts if p.get("prompt_id") == args.prompt_id), None)
    if not entry:
        print(f"No prompt_id {args.prompt_id!r} in manifest", file=sys.stderr)
        return 1
    ver = str(entry.get("version", "0.0.0"))
    cur = _major_minor_tuple(ver)

    if not baseline_path.is_file():
        print(f"Missing baseline file: {baseline_path}", file=sys.stderr)
        return 1
    base_raw = baseline_path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    base = _major_minor_tuple(base_raw)

    if cur > base:
        print(
            f"Prompt {args.prompt_id!r} version {ver!r} (major.minor {cur[0]}.{cur[1]}) is "
            f"ahead of stress gate baseline {base_raw!r} ({base[0]}.{base[1]}). "
            f"Run stress evals, then bump {baseline_path}",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: manifest {ver!r} (major.minor {cur[0]}.{cur[1]}) "
        f"<= baseline {base_raw!r} ({base[0]}.{base[1]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
