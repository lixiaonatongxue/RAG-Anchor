"""Deterministic, traceable retrieval over persisted PDF chunks.

The service is deliberately minimal:

* it reads the chunk rows produced by :class:`DocumentIndexService` from
  SQLite; it never re-parses the PDF,
* matching is plain Python substring matching (case-sensitive by default),
* every hit carries the document, chunk, page and *text* offsets that existed
  when the chunk was persisted,
* no embeddings, vector store, LLM or extra dependency is involved.

Offsets are Python ``str`` slice boundaries into the stored page text. They are
**not** PDF user-space coordinates and **not** PDFium character indices. This
service never fabricates rectangles or page geometry. Callers that need geometry
for an exact retrieved span must pass ``document_id`` + ``page_index`` +
``match_start`` + ``match_end`` to
:class:`rag_anchor.services.evidence_service.EvidenceLocationService`
(``POST /api/v1/documents/{document_id}/evidence/locate``), or use the existing
anchor search API. The evidence service maps Python offsets to PDFium character
indices explicitly and only ever returns real PDFium rectangles.

Candidate collection, de-duplication and overlap selection
----------------------------------------------------------
Every occurrence inside every persisted chunk is collected first, including
occurrences that overlap each other inside one chunk. A match is then mapped to
absolute page-text offsets (``chunk.start_offset + local_start``). If the same
absolute span appears in several overlapping chunks, it is reported once and
the evidence (``chunk_id``, ``chunk_text``, ``chunk_index``) is taken from the
chunk with the lowest ``chunk_index``.

After de-duplication, a single left-to-right pass per page selects the
non-overlapping matches. Overlap is decided on absolute page-text offsets, not
on chunk-local offsets, so a match skipped in one chunk cannot suppress a
later, distinct match in another chunk.

Queries are never matched across chunk boundaries: this service does not
concatenate chunks. A query that only matches by joining two chunks is not
returned.

Final ordering is deterministic:
``(document_id, page_index, match_start, match_end, chunk_index, chunk_id)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from rag_anchor.infra.database import DocumentRepository
from rag_anchor.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
)

MAX_QUERY_LENGTH = 2000
"""Maximum accepted query length in Python characters."""

DEFAULT_RETRIEVAL_LIMIT = 20
MAX_RETRIEVAL_LIMIT = 100


@dataclass(frozen=True, slots=True)
class RetrievalMatch:
    """One traceable occurrence of the query in a persisted chunk."""

    document_id: UUID
    chunk_id: str
    page_index: int
    page_number: int
    chunk_index: int
    start_offset: int
    end_offset: int
    match_start: int
    match_end: int
    hit_text: str
    chunk_text: str
    extraction_version: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A page of matches plus the unpaginated total."""

    query: str
    match_case: bool
    document_id: UUID | None
    total: int
    limit: int
    offset: int
    matches: list[RetrievalMatch]


class ChunkRetrievalService:
    """Search persisted chunks while preserving full text provenance."""

    def __init__(self, document_service: DocumentService) -> None:
        self._document_service = document_service
        self._repository = document_service.repository

    @property
    def repository(self) -> DocumentRepository:
        return self._repository

    def search_chunks(
        self,
        query: str,
        *,
        document_id: UUID | None = None,
        match_case: bool = True,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        offset: int = 0,
    ) -> RetrievalResult:
        """Return page-level non-overlapping occurrences of ``query``.

        Candidates are collected from each chunk, de-duplicated by absolute
        page-text span (lowest ``chunk_index`` wins), and then filtered by a
        page-level left-to-right non-overlap pass. Chunks are never
        concatenated for matching.

        Parameters
        ----------
        query:
            Non-empty, non-whitespace search text. Matching is literal; no
            whitespace normalisation or tokenisation is applied.
        document_id:
            Optional document filter. An unknown ID raises
            :class:`DocumentNotFoundError` instead of silently returning no
            results.
        match_case:
            ``True`` (default) for case-sensitive matching. ``False`` compares
            same-length windows using ``str.lower()``. This deliberately does
            not implement length-changing Unicode case folding: for example,
            ``"ß"`` and ``"SS"`` are not treated as equivalent. Offsets always
            refer to the original text.
        limit:
            Maximum number of matches returned. ``1..MAX_RETRIEVAL_LIMIT``.
        offset:
            Number of matches skipped after de-duplication and page-level
            overlap selection.

        Raises
        ------
        TypeError
            If ``query`` is not a string or a numeric argument is not an int.
        ValueError
            If ``query`` is empty/whitespace-only or too long, or if
            ``limit``/``offset`` are out of range.
        DocumentNotFoundError
            If ``document_id`` is provided but has no metadata row.
        """
        _validate_query(query)
        _validate_limit(limit)
        _validate_offset(offset)

        if document_id is not None and self._document_service.get_document(document_id) is None:
            raise DocumentNotFoundError("Document not found")

        candidates: list[RetrievalMatch] = []
        for chunk in self._repository.list_chunks(document_id=document_id):
            for local_start in _find_all_occurrences(chunk.text, query, match_case):
                match_start = chunk.start_offset + local_start
                match_end = match_start + len(query)
                candidates.append(
                    RetrievalMatch(
                        document_id=UUID(chunk.document_id),
                        chunk_id=chunk.chunk_id,
                        page_index=chunk.page_index,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        start_offset=chunk.start_offset,
                        end_offset=chunk.end_offset,
                        match_start=match_start,
                        match_end=match_end,
                        hit_text=chunk.text[local_start : local_start + len(query)],
                        chunk_text=chunk.text,
                        extraction_version=chunk.extraction_version,
                    )
                )

        deduplicated = _deduplicate(candidates)
        selected = _select_page_non_overlapping(deduplicated)

        return RetrievalResult(
            query=query,
            match_case=match_case,
            document_id=document_id,
            total=len(selected),
            limit=limit,
            offset=offset,
            matches=selected[offset : offset + limit],
        )


def _validate_query(query: object) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")


def _validate_limit(limit: object) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    if limit < 1 or limit > MAX_RETRIEVAL_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_RETRIEVAL_LIMIT}")


def _validate_offset(offset: object) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise TypeError("offset must be an integer")
    if offset < 0:
        raise ValueError("offset must be zero or greater")


def _find_all_occurrences(text: str, query: str, match_case: bool) -> list[int]:
    """Return *every* match start in ``text``, including overlapping ones.

    The cursor always advances by one character so ``"aa"`` in ``"aaaa"``
    yields ``0, 1, 2``. Non-overlap is a page-level decision applied later,
    after candidates from overlapping chunks have been merged; deciding it
    per chunk would let one chunk's cursor hide a global match. Keeping this
    enumeration exhaustive also means the same absolute span is observed from
    every chunk that contains it, which is exactly what cross-chunk
    de-duplication needs.

    With ``match_case=False`` every window of ``len(query)`` characters is
    compared after ``str.lower()``. This is intentionally *not* Unicode case
    folding: conversions that change length (for example ``"ß"`` vs ``"SS"``)
    are not treated as equivalent, and offsets always refer to the original
    text.
    """
    width = len(query)
    occurrences: list[int] = []

    if match_case:
        start = 0
        while True:
            index = text.find(query, start)
            if index < 0:
                break
            occurrences.append(index)
            start = index + 1
        return occurrences

    folded_query = query.lower()
    start = 0
    while start + width <= len(text):
        if text[start : start + width].lower() == folded_query:
            occurrences.append(start)
        start += 1
    return occurrences


def _ranking_key(match: RetrievalMatch) -> tuple[str, int, int, int, int, str]:
    """Deterministic page-offset-first ordering, independent of row order."""
    return (
        str(match.document_id),
        match.page_index,
        match.match_start,
        match.match_end,
        match.chunk_index,
        match.chunk_id,
    )


def _deduplicate(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """Collapse identical absolute page-text spans found in overlapping chunks.

    The chunk with the lowest ``chunk_index`` wins. Sorting happens before the
    scan so the result is independent of SQLite row order, and ``chunk_id``
    breaks ties for a deterministic attribution.
    """
    ordered = sorted(matches, key=_ranking_key)
    deduplicated: list[RetrievalMatch] = []
    previous_key: tuple[str, int, int, int] | None = None
    for match in ordered:
        key = (str(match.document_id), match.page_index, match.match_start, match.match_end)
        if key == previous_key:
            continue
        deduplicated.append(match)
        previous_key = key
    return deduplicated


def _select_page_non_overlapping(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """Keep the left-to-right non-overlapping matches of each page.

    ``matches`` must already be de-duplicated and sorted by
    :func:`_ranking_key`. Overlap is evaluated on absolute page-text offsets,
    so a hit that overlaps an earlier one is dropped even when the two hits
    come from different chunks. Because the candidates are ordered by
    ``match_start``, the first hit in an overlapping group always wins, and
    a later, distinct hit (``match_start >= last_end``) is never dropped.
    """
    selected: list[RetrievalMatch] = []
    page_key: tuple[str, int] | None = None
    last_end = 0

    for match in matches:
        current_page = (str(match.document_id), match.page_index)
        if current_page != page_key:
            page_key = current_page
            last_end = 0
        if match.match_start >= last_end:
            selected.append(match)
            last_end = match.match_end
    return selected
