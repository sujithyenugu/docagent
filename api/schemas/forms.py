"""Pydantic v2 schemas for the /api/v1/forms/fill endpoint."""

from typing import Any

from pydantic import BaseModel, Field


class FormFillResponse(BaseModel):
    fields: dict[str, Any] = Field(
        description="Extracted field values keyed by field name. "
        "Null values indicate the field was not found."
    )
    field_sources: dict[str, str] = Field(
        description="Maps each field name to the filename it was extracted from."
    )
    missing_fields: list[str] = Field(
        description="Fields required by the template that could not be extracted."
    )
    request_id: str
