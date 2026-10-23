"""Pydantic models exposed by the document and anchor search API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_anchor.services.qa_service import (
    DEFAULT_MAX_CITATIONS,
    MAX_CITATIONS,
    MAX_QUESTION_LENGTH,
)
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


class EvidenceSpanRequest(BaseModel):
    """One exact page-text span whose PDF geometry is requested.

    ``match_start``/``match_end`` are Python string offsets into the stored
    page text, never PDFium character indices and never PDF coordinates. They
    identify the occurrence; ``hit_text`` is echoed back and verified, it is
    never used to search for the occurrence.
    """

    page_index: int = Field(ge=0)
    match_start: int = Field(ge=0)
    match_end: int = Field(ge=1)
    hit_text: str = Field(min_length=1)
    chunk_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _span_must_be_consistent(self) -> EvidenceSpanRequest:
        if self.match_end <= self.match_start:
            raise ValueError("match_end must be greater than match_start")
        if len(self.hit_text) != self.match_end - self.match_start:
            raise ValueError("hit_text length must equal match_end - match_start")
        return self


class EvidenceLocateRequest(BaseModel):
    """Batch of spans to locate in one stored document.

    Items keep their order and are never deduplicated. The whole request
    fails if any item is invalid.
    """

    items: list[EvidenceSpanRequest] = Field(min_length=1, max_length=20)


class LocatedEvidenceResponse(BaseModel):
    """One span located in the rendered PDF.

    ``char_start``/``char_count`` are PDFium character indices.
    ``rects``/``bbox`` are absolute PDF user space rectangles in points.
    """

    model_config = ConfigDict(from_attributes=True)

    page_index: int
    page_number: int
    match_start: int
    match_end: int
    hit_text: str
    chunk_id: str | None
    char_start: int
    char_count: int
    matched_text: str
    rects: list[RectResponse]
    bbox: RectResponse | None
    page: PageResponse


class EvidenceLocateResponse(BaseModel):
    """All located spans for one request, in request order."""

    document_id: UUID
    total: int = Field(ge=0)
    results: list[LocatedEvidenceResponse]


class QuestionRequest(BaseModel):
    """One question scoped to an indexed document.

    Retrieval is always literal; no semantic interpretation is performed.
    ``answer_mode`` defaults to ``evidence_only``, preserving the existing
    no-model behavior. ``grounded_generation`` only calls a model when the
    server is explicitly configured and enabled.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    retrieval_query: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)
    match_case: bool = False
    max_citations: int = Field(default=DEFAULT_MAX_CITATIONS, ge=1, le=MAX_CITATIONS)
    answer_mode: Literal["evidence_only", "grounded_generation"] = "evidence_only"

    @field_validator("question")
    @classmethod
    def _question_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be a non-empty, non-whitespace string")
        return value

    @field_validator("retrieval_query")
    @classmethod
    def _retrieval_query_must_contain_text(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("retrieval_query must be a non-empty, non-whitespace string")
        return value


class EvidenceCitationResponse(BaseModel):
    """One retrieved page-text span, ready for the evidence locate endpoint.

    ``match_start``/``match_end`` are Python string offsets into the stored
    page text, exactly as returned by chunk retrieval. No PDF coordinates are
    produced by question answering.
    """

    model_config = ConfigDict(from_attributes=True)

    citation_id: str
    document_id: UUID
    chunk_id: str
    page_index: int = Field(ge=0)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    match_start: int = Field(ge=0)
    match_end: int = Field(ge=1)
    hit_text: str
    chunk_text: str
    extraction_version: str


class QuestionResponse(BaseModel):
    """Answer plus independently verifiable citations.

    In ``evidence_only`` mode ``answer`` is a fixed server notice. In
    ``grounded_generation`` mode it is model text whose citation IDs were
    validated by the server; that does not prove every sentence is supported.
    Every citation itself is reconstructed from the persisted chunk index of
    ``document_id``, never from model output.
    """

    document_id: UUID
    question: str
    retrieval_query: str
    match_case: bool
    answer_mode: Literal["evidence_only", "grounded_generation"]
    status: Literal["evidence_found", "insufficient_evidence", "answered"]
    answer: str
    total_matches: int = Field(ge=0)
    citations: list[EvidenceCitationResponse]
