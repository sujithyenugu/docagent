"""Pydantic v2 schemas for the /api/v1/extract endpoint."""

from pydantic import BaseModel, Field


class ExtractionMetadata(BaseModel):
    page_count: int
    duration_ms: int
    model: str
    backend: str  # "local" | "cloud"


class ExtractResponse(BaseModel):
    markdown: str
    metadata: ExtractionMetadata
    request_id: str = Field(description="Echoed from X-Request-ID or server-generated UUID")
