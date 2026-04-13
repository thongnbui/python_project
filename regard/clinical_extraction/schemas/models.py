"""Pydantic models aligned with extraction_output.json."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """Evidence link for an extracted fact."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    quote: str


class ExtractionMeta(BaseModel):
    """Run metadata merged into model output for auditing."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: str
    model: str
    warnings: list[str] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    """Structured extraction result validated after every LLM call."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    entities: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    insufficient_context: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    meta: ExtractionMeta = Field(alias="_meta")


def validate_extraction_json(data: str | bytes | dict[str, Any]) -> ExtractionOutput:
    """Parse and validate a model response."""
    if isinstance(data, (str, bytes)):
        return ExtractionOutput.model_validate_json(data)
    return ExtractionOutput.model_validate(data)
