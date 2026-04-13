"""Golden JSON parsing for ExtractionOutput."""

from __future__ import annotations

import json

import pytest

from regard.clinical_extraction.schemas.models import (
    ExtractionOutput,
    JudgeOutput,
    validate_extraction_json,
)


def test_valid_roundtrip() -> None:
    payload = {
        "entities": [{"type": "problem", "value": "hypertension"}],
        "citations": [{"chunk_id": "excerpt", "quote": "HTN"}],
        "insufficient_context": False,
        "clarifying_questions": [],
        "_meta": {"prompt_version": "t", "model": "m"},
    }
    m = validate_extraction_json(payload)
    dumped = m.model_dump(mode="json", by_alias=True)
    assert dumped["_meta"]["model"] == "m"


def test_validate_json_string() -> None:
    raw = json.dumps(
        {
            "entities": [],
            "citations": [],
            "insufficient_context": True,
            "clarifying_questions": ["need more"],
            "_meta": {"prompt_version": "t", "model": "m"},
        }
    )
    validate_extraction_json(raw)


def test_judge_output_valid() -> None:
    jo = JudgeOutput.model_validate(
        {
            "score": 2,
            "rationale": "Entities traceable to excerpt.",
            "unsupported_claims": [],
        }
    )
    assert jo.score == 2


def test_reject_extra_top_level_key() -> None:
    bad = {
        "entities": [],
        "citations": [],
        "insufficient_context": False,
        "clarifying_questions": [],
        "_meta": {"prompt_version": "t", "model": "m"},
        "unexpected": 1,
    }
    with pytest.raises(Exception):
        ExtractionOutput.model_validate(bad)
