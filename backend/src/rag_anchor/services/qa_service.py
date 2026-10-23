"""Evidence-only question answering over persisted chunks.

No language model is involved in this MVP. The service runs the existing
literal chunk retrieval scoped to one document and returns traceable citations
plus a fixed, explicitly labelled notice. It never states a factual conclusion
that is absent from the returned citations, and it never returns PDF geometry;
callers must use the existing evidence location endpoint for a citation.

The retrieval query is literal. A natural-language question is not interpreted
semantically; callers may supply a shorter literal ``retrieval_query`` from the
document. When no citation is found the service refuses to answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from rag_anchor.domain.text import TEXT_EXTRACTION_VERSION
from rag_anchor.infra.database import DocumentRepository
from rag_anchor.services.document_service import DocumentNotFoundError, DocumentService
from rag_anchor.services.retrieval_service import (
    MAX_QUERY_LENGTH,
    ChunkRetrievalService,
    RetrievalMatch,
)

ANSWER_MODE_EVIDENCE_ONLY = "evidence_only"
"""The only answer mode in this MVP: retrieved evidence, no generated answer."""

STATUS_EVIDENCE_FOUND = "evidence_found"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

DEFAULT_MAX_CITATIONS = 5
MAX_CITATIONS = 20
MAX_QUESTION_LENGTH = MAX_QUERY_LENGTH
MAX_QUERY_DISPLAY_LENGTH = 120

INSUFFICIENT_EVIDENCE_ANSWER = (
    "当前为证据展示模式(未接入语言模型)。"
    "没有在文档中检索到与「{query}」字面匹配的原文证据,"
    "因此无法依据该文档回答。请尝试文档中出现的原文词句。"
)

EVIDENCE_FOUND_ANSWER = (
    "当前为证据展示模式(未接入语言模型)。"
    "按字面「{query}」检索到 {total} 条原文证据,以下返回 {returned} 条引用。"
    "系统不生成结论,请点击引用核对 PDF 原文。"
)


class QuestionIndexUnavailableError(RuntimeError):
    """The document has no usable index for question answering (HTTP 409)."""


class QuestionCitationScopeError(RuntimeError):
    """Retrieval returned a citation outside the requested document."""


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    """One retrieved source span, ready for evidence/locate on click."""

    citation_id: str
    document_id: UUID
    chunk_id: str
    page_index: int
    page_number: int
    chunk_index: int
    match_start: int
    match_end: int
    hit_text: str
    chunk_text: str
    extraction_version: str


@dataclass(frozen=True, slots=True)
class QuestionAnswer:
    """The evidence-only result of one question."""

    document_id: UUID
    question: str
    retrieval_query: str
    match_case: bool
    answer_mode: str
    status: str
    answer: str
    total_matches: int
    citations: list[EvidenceCitation]


class QuestionAnswerService:
    """Answer questions with literal citations from one indexed document."""

    def __init__(
        self,
        document_service: DocumentService,
        retrieval_service: ChunkRetrievalService,
    ) -> None:
        self._document_service = document_service
        self._retrieval_service = retrieval_service
        self._repository: DocumentRepository = document_service.repository

    @property
    def repository(self) -> DocumentRepository:
        return self._repository

    def answer(
        self,
        document_id: UUID,
        question: str,
        *,
        retrieval_query: str | None = None,
        match_case: bool = False,
        max_citations: int = DEFAULT_MAX_CITATIONS,
    ) -> QuestionAnswer:
        """Return citations for the question, or an explicit refusal.

        ``retrieval_query`` falls back to the question itself. Matching is the
        existing literal chunk search; no semantic interpretation is performed.
        Only the first ``max_citations`` matches become citations, while
        ``total_matches`` reports the unpaginated count.
        """
        _validate_text(question, "question", MAX_QUESTION_LENGTH)
        effective_query = question if retrieval_query is None else retrieval_query
        _validate_text(effective_query, "retrieval_query", MAX_QUERY_LENGTH)
        _validate_max_citations(max_citations)

        if self._document_service.get_document(document_id) is None:
            raise DocumentNotFoundError("Document not found")

        metadata = self._repository.get_document_index_metadata(document_id)
        if metadata is None:
            raise QuestionIndexUnavailableError("Document has not been indexed")
        if metadata.extraction_version != TEXT_EXTRACTION_VERSION:
            raise QuestionIndexUnavailableError(
                "Stored index was built with a different extraction version"
            )

        result = self._retrieval_service.search_chunks(
            effective_query,
            document_id=document_id,
            match_case=match_case,
            limit=max_citations,
            offset=0,
        )

        citations = self._build_citations(document_id, result.matches)
        if not citations:
            return QuestionAnswer(
                document_id=document_id,
                question=question,
                retrieval_query=effective_query,
                match_case=match_case,
                answer_mode=ANSWER_MODE_EVIDENCE_ONLY,
                status=STATUS_INSUFFICIENT_EVIDENCE,
                answer=INSUFFICIENT_EVIDENCE_ANSWER.format(query=_display_query(effective_query)),
                total_matches=result.total,
                citations=[],
            )

        return QuestionAnswer(
            document_id=document_id,
            question=question,
            retrieval_query=effective_query,
            match_case=match_case,
            answer_mode=ANSWER_MODE_EVIDENCE_ONLY,
            status=STATUS_EVIDENCE_FOUND,
            answer=EVIDENCE_FOUND_ANSWER.format(
                query=_display_query(effective_query),
                total=result.total,
                returned=len(citations),
            ),
            total_matches=result.total,
            citations=citations,
        )

    def _build_citations(
        self,
        document_id: UUID,
        matches: list[RetrievalMatch],
    ) -> list[EvidenceCitation]:
        citations: list[EvidenceCitation] = []
        for index, match in enumerate(matches, start=1):
            if match.document_id != document_id:
                # Retrieval is document-scoped, so this can only signal a bug.
                # Never leak a citation from another document.
                raise QuestionCitationScopeError(
                    "Retrieval returned a citation from another document"
                )
            citations.append(
                EvidenceCitation(
                    citation_id=f"E{index}",
                    document_id=match.document_id,
                    chunk_id=match.chunk_id,
                    page_index=match.page_index,
                    page_number=match.page_number,
                    chunk_index=match.chunk_index,
                    match_start=match.match_start,
                    match_end=match.match_end,
                    hit_text=match.hit_text,
                    chunk_text=match.chunk_text,
                    extraction_version=match.extraction_version,
                )
            )
        return citations


def _display_query(query: str) -> str:
    if len(query) <= MAX_QUERY_DISPLAY_LENGTH:
        return query
    return f"{query[: MAX_QUERY_DISPLAY_LENGTH - 1]}…"


def _validate_text(value: str, name: str, max_length: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty, non-whitespace string")
    if len(value) > max_length:
        raise ValueError(f"{name} must be at most {max_length} characters")


def _validate_max_citations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_citations must be an integer")
    if not 1 <= value <= MAX_CITATIONS:
        raise ValueError(f"max_citations must be between 1 and {MAX_CITATIONS}")


__all__ = [
    "ANSWER_MODE_EVIDENCE_ONLY",
    "DEFAULT_MAX_CITATIONS",
    "EVIDENCE_FOUND_ANSWER",
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "MAX_CITATIONS",
    "MAX_QUESTION_LENGTH",
    "STATUS_EVIDENCE_FOUND",
    "STATUS_INSUFFICIENT_EVIDENCE",
    "EvidenceCitation",
    "QuestionAnswer",
    "QuestionAnswerService",
    "QuestionCitationScopeError",
    "QuestionIndexUnavailableError",
]
