"""Smoke tests for agents/chat_cli.py (offline only; no API)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FEATURE = Path(__file__).resolve().parents[1]
CLI = FEATURE / "agents" / "chat_cli.py"
EXAMPLE = FEATURE / "agents" / "chat_example.txt"


def test_chat_cli_offline_script() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--offline",
            "--script",
            str(EXAMPLE),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "retrieve_chart" in proc.stdout
    assert "draft_progress_note" in proc.stdout


def test_chat_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(CLI), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    assert proc.returncode == 0
    assert "offline" in proc.stdout.lower() or "offline" in proc.stderr.lower()
