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


class SemanticIndexResponse(BaseModel):
    """Statistics of one completed derived semantic index build.

    ``source_fingerprint`` identifies the text index the vectors were built
    from. ``model_name``/``model_version``/``model_digest``/``dimension``/
    ``normalization`` identify the embedding used; a later query is rejected
    when the active provider no longer matches them. ``index_version`` and
    ``candidate_version`` record the persisted layout and split rules, so an
    index produced by an older build is never treated as compatible.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    model_name: str
    model_version: str
    model_digest: str
    index_version: int = Field(ge=0)
    candidate_version: int = Field(ge=0)
    dimension: int = Field(gt=0)
    normalization: Literal["l2"]
    source_fingerprint: str
    candidate_count: int = Field(ge=0)
    built_at: datetime


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

    ``retrieval_mode`` defaults to ``literal`` and preserves the M1 behavior
    exactly: substring matching over persisted chunks. ``semantic`` embeds the
    question itself (``retrieval_query`` is ignored) and ranks derived
    candidates; it requires a separately built semantic index and never falls
    back to literal matching. ``answer_mode`` defaults to ``evidence_only``,
    preserving the existing no-model behavior. ``grounded_generation`` only
    calls a model when the server is explicitly configured and enabled.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    retrieval_query: str | None = Field(default=None, max_length=MAX_QUERY_LENGTH)
    retrieval_mode: Literal["literal", "semantic"] = "literal"
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
    page text. For a literal citation they delimit the keyword occurrence; for
    a semantic citation they delimit the ranked candidate span. They are never
    PDF coordinates, and question answering never produces geometry.

    ``retrieval_kind`` makes the difference explicit. A ``semantic`` citation
    carries a cosine ``score`` and ``support_kind == "semantic_candidate"``;
    its span is the candidate itself, not a keyword match. A ``literal``
    citation keeps the M1 contract: ``support_*`` covers the complete sentence
    containing the hit (``support_kind == "sentence"``) or exactly the hit
    (``support_kind == "retrieval_match"``).

    Both kinds are exact slices of the stored page text and stay inside the
    citation's chunk; ``support_kind`` never claims factual support for a
    generated answer.
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
    retrieval_kind: Literal["literal", "semantic"] = "literal"
    score: float | None = None
    support_text: str = Field(min_length=1)
    support_start: int = Field(ge=0)
    support_end: int = Field(ge=1)
    support_kind: Literal["sentence", "retrieval_match", "semantic_candidate"]

    @model_validator(mode="after")
    def _support_span_must_contain_the_hit(self) -> EvidenceCitationResponse:
        if self.support_end <= self.support_start:
            raise ValueError("support_end must be greater than support_start")
        if len(self.support_text) != self.support_end - self.support_start:
            raise ValueError("support_text length must equal support_end - support_start")
        if self.support_start > self.match_start or self.support_end < self.match_end:
            raise ValueError("support span must contain the retrieval hit")
        if self.retrieval_kind == "semantic":
            if self.support_kind != "semantic_candidate":
                raise ValueError("semantic citations must use a semantic candidate support")
            if self.score is None:
                raise ValueError("semantic citations must include a similarity score")
            if self.support_start != self.match_start or self.support_end != self.match_end:
                raise ValueError("semantic support span must equal the candidate span")
        else:
            if self.support_kind == "semantic_candidate":
                raise ValueError("literal citations cannot use a semantic candidate support")
            if self.score is not None:
                raise ValueError("literal citations cannot include a semantic score")
        return self


class QuestionResponse(BaseModel):
    """Answer plus independently verifiable citations.

    ``retrieval_mode`` echoes the requested layer. Literal citations are
    keyword occurrences; semantic citations are ranked candidate spans and
    carry a cosine score. In ``evidence_only`` mode ``answer`` is a fixed
    server notice. In ``grounded_generation`` mode it is model text whose
    citation IDs were validated by the server; that does not prove every
    sentence is supported. Every citation itself is reconstructed from the
    persisted index of ``document_id``, never from model output.
    """

    document_id: UUID
    question: str
    retrieval_query: str
    retrieval_mode: Literal["literal", "semantic"]
    match_case: bool
    answer_mode: Literal["evidence_only", "grounded_generation"]
    status: Literal["evidence_found", "insufficient_evidence", "answered"]
    answer: str
    total_matches: int = Field(ge=0)
    citations: list[EvidenceCitationResponse]
