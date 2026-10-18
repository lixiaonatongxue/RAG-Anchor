"""Pydantic models exposed by the document and anchor search API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentResponse(BaseModel):
    """Public metadata for one stored PDF.

    The server-side absolute path is intentionally absent.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    filename: str
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=1)
    created_at: datetime


class DocumentListResponse(BaseModel):
    """One page of document metadata."""

    items: list[DocumentResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class AnchorSearchRequest(BaseModel):
    """Search payload for one stored document."""

    query: str
    match_case: bool = True

    @field_validator("query")
    @classmethod
    def _query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must be a non-empty, non-whitespace string")
        return value


class RectResponse(BaseModel):
    """Axis-aligned rectangle in absolute PDF user space points."""

    model_config = ConfigDict(from_attributes=True)

    left: float
    bottom: float
    right: float
    top: float


class PageResponse(BaseModel):
    """Page geometry needed to interpret hit rectangles."""

    model_config = ConfigDict(from_attributes=True)

    page_index: int
    width_pt: float
    height_pt: float
    rotation: int
    cropbox: tuple[float, float, float, float]


class AnchorHitResponse(BaseModel):
    """One occurrence of the query in the document."""

    model_config = ConfigDict(from_attributes=True)

    page_index: int
    page_number: int
    matched_text: str
    char_start: int
    char_count: int
    rects: list[RectResponse]
    bbox: RectResponse | None
    page: PageResponse


class AnchorSearchResponse(BaseModel):
    """All matches for one query, in document reading order.

    Coordinates are absolute PDF user space points. The API layer never
    re-derives rectangles, subtracts the CropBox origin, or applies rotation.
    """

    document_id: UUID
    query: str
    match_case: bool
    total: int = Field(ge=0)
    results: list[AnchorHitResponse]
