"""Pydantic models exposed by the document and anchor search API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_anchor.services.retrieval_service import (
    DEFAULT_RETRIEVAL_LIMIT,
    MAX_QUERY_LENGTH,
    MAX_RETRIEVAL_LIMIT,
)


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


class DocumentIndexResponse(BaseModel):
    """Statistics of one completed index build.

    ``indexed_at`` is the wall-clock time of this run. Repeated indexing of an
    unchanged PDF rebuilds the same rows and returns the same counts; it is not
    a cache hit.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    fingerprint: str
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    filtered_chunk_count: int = Field(ge=0)
    indexed_at: datetime


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


class ChunkRetrievalRequest(BaseModel):
    """Search persisted chunks across all documents or one document."""

    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    document_id: UUID | None = None
    match_case: bool = True
    limit: int = Field(
        default=DEFAULT_RETRIEVAL_LIMIT,
        ge=1,
        le=MAX_RETRIEVAL_LIMIT,
    )
    offset: int = Field(default=0, ge=0)

    @field_validator("query")
    @classmethod
    def _query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must be a non-empty, non-whitespace string")
        return value


class RetrievalHitResponse(BaseModel):
    """One traceable match in a persisted chunk.

    ``start_offset``/``end_offset`` delimit the chunk inside its stored page
    text. ``match_start``/``match_end`` delimit the hit inside that same page
    text. These are Python string offsets, never PDF user-space coordinates.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    chunk_id: str
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    match_start: int = Field(ge=0)
    match_end: int = Field(ge=0)
    hit_text: str
    chunk_text: str
    extraction_version: str


class ChunkRetrievalResponse(BaseModel):
    """One page of retrieval matches plus the unpaginated total."""

    query: str
    match_case: bool
    document_id: UUID | None
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    results: list[RetrievalHitResponse]
