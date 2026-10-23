"""Exact, read-only evidence location over a persisted PDF index.

This service turns a *page-text span* produced by the chunk index into real
PDFium character rectangles. It never searches the PDF for the span text: the
request already identifies the occurrence by ``page_index`` + ``match_start`` +
``match_end``, so no occurrence can be guessed from a repeated ``hit_text``.

Three distinct spaces are involved and must never be mixed:

* **Python ``str`` offsets** (``match_start`` / ``match_end``) index the stored
  page text, i.e. the string returned by PDFium's ``get_text_range()``.
* **PDFium character indices** (``char_start`` / ``char_count``) index the
  engine's internal character array, whose size may differ from
  ``len(page_text)`` when supplementary-plane characters are present.
* **PDF user-space geometry** (``rects`` / ``bbox``) is expressed in absolute
  PDF points.

The mapping is built once per page from the per-character strings returned by
``get_text_range(index, 1)``. Empty segments (characters PDFium cannot express
in Python, notably non-BMP code points) make the cumulative offsets
non-strictly increasing; the mapping therefore resolves the *exact* Python span
to the smallest PDFium range that reproduces it, and fails explicitly when a
boundary falls inside a character that cannot be expressed. An empty segment
strictly *inside* the resolved range is rejected as well: PDFium still reports
a real character box for such a character, so including it would highlight a
glyph the caller never asked for. Only empty segments before the start or after
the end are excluded, exactly by boundary selection.

Index compatibility is verified read-only. The stored PDF SHA-256, the stored
page text and the re-extracted page text must match; a different installed
PDFium build is tolerated only because every involved page is re-extracted and
compared exactly. Nothing is re-indexed, no database row is written and the
stored fingerprint is never refreshed.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from pypdfium2 import PdfDocument, PdfiumError, PdfPage, PdfTextPage

from rag_anchor.domain.models import PageDescriptor, PdfRect
from rag_anchor.domain.text import TEXT_EXTRACTION_VERSION, PageText
from rag_anchor.infra.database import DocumentRepository
from rag_anchor.services.anchor_service import build_page_descriptor, match_rects
from rag_anchor.services.document_service import DocumentNotFoundError, DocumentService
from rag_anchor.services.index_service import detect_pdfium_version, hash_file

logger = logging.getLogger(__name__)

EVIDENCE_ITEMS_MIN = 1
"""Minimum number of spans accepted by one location request."""

EVIDENCE_ITEMS_MAX = 20
"""Maximum number of spans accepted by one location request."""


class EvidenceLocationError(RuntimeError):
    """Base class for evidence location failures."""


class EvidenceDocumentNotFoundError(EvidenceLocationError):
    """The document has no metadata row (HTTP 404)."""


class EvidenceValidationError(EvidenceLocationError):
    """The request is invalid against the stored page text (HTTP 422)."""


class EvidenceIndexUnavailableError(EvidenceLocationError):
    """The document has no usable index (HTTP 409)."""


class EvidenceIndexMismatchError(EvidenceLocationError):
    """The stored index no longer matches the current PDF (HTTP 409)."""


class EvidenceMappingError(EvidenceLocationError):
    """A span cannot be mapped to an exact PDFium range (HTTP 409)."""


class EvidenceSourceError(EvidenceLocationError):
    """The stored PDF is missing, unreadable or unusable (HTTP 500)."""


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """One requested span of stored page text.

    ``match_start`` and ``match_end`` are Python ``str`` slice boundaries into
    the stored page text, exactly as returned by chunk retrieval. ``hit_text``
    is redundant on purpose: the service verifies it against the stored text.
    """

    page_index: int
    match_start: int
    match_end: int
    hit_text: str
    chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class LocatedEvidence:
    """One span located in the rendered PDF."""

    document_id: UUID
    page_index: int
    page_number: int
    match_start: int
    match_end: int
    hit_text: str
    chunk_id: str | None
    char_start: int
    char_count: int
    matched_text: str
    rects: tuple[PdfRect, ...]
    page: PageDescriptor

    @property
    def bbox(self) -> PdfRect | None:
        """Smallest rectangle enclosing every real line rectangle."""
        if not self.rects:
            return None
        return PdfRect(
            left=min(rect.left for rect in self.rects),
            bottom=min(rect.bottom for rect in self.rects),
            right=max(rect.right for rect in self.rects),
            top=max(rect.top for rect in self.rects),
        )


def build_offset_map(text_page: PdfTextPage) -> tuple[list[str], list[int]]:
    """Return per-character strings and cumulative Python offsets.

    ``segments[i]`` is what PDFium returns for character ``i``. It may be the
    empty string when the engine's character cannot be represented in Python
    (for example a supplementary-plane code point). ``offsets`` always has
    ``len(segments) + 1`` entries; ``offsets[0] == 0`` and
    ``offsets[-1] == len("".join(segments))``.
    """
    count = text_page.count_chars()
    segments: list[str] = []
    offsets: list[int] = [0]
    total = 0
    for index in range(count):
        segment = text_page.get_text_range(index, 1)
        segments.append(segment)
        total += len(segment)
        offsets.append(total)
    return segments, offsets


def map_python_span(
    offsets: Sequence[int],
    match_start: int,
    match_end: int,
) -> tuple[int, int]:
    """Map a Python span to the PDFium character range ``[start, end)``.

    The start boundary is resolved to the *last* index whose cumulative offset
    equals ``match_start`` so that leading empty characters are excluded. The
    end boundary is resolved to the *first* index whose cumulative offset
    equals ``match_end`` so that trailing empty characters are excluded.

    Raises ``EvidenceValidationError`` when the span is outside the text and
    ``EvidenceMappingError`` when a boundary falls inside a character that
    produces more than one Python character (or otherwise has no exact
    boundary), or when an unrepresentable character (an empty segment) lies
    strictly inside the resolved range. No approximate range is ever returned.
    """
    if match_start < 0:
        raise EvidenceValidationError("match_start must be zero or greater")
    if match_end <= match_start:
        raise EvidenceValidationError("match_end must be greater than match_start")
    if not offsets:
        raise EvidenceMappingError("empty offset map")
    if match_end > offsets[-1]:
        raise EvidenceValidationError("span is outside the stored page text")

    start_index = bisect_right(offsets, match_start) - 1
    if start_index < 0 or offsets[start_index] != match_start:
        raise EvidenceMappingError(
            "match_start falls inside a character PDFium cannot express exactly"
        )

    end_index = bisect_left(offsets, match_end)
    if end_index >= len(offsets) or offsets[end_index] != match_end:
        raise EvidenceMappingError(
            "match_end falls inside a character PDFium cannot express exactly"
        )
    if end_index <= start_index:
        raise EvidenceMappingError("span cannot be mapped to a non-empty character range")
    if any(offsets[index] == offsets[index + 1] for index in range(start_index, end_index)):
        # An empty segment inside the range means PDFium still has a character
        # there, but it contributes no Python text. Its real character box
        # would be highlighted even though the caller never asked for it.
        raise EvidenceMappingError(
            "span contains a PDFium character that produces no text; "
            "exact geometry cannot be proven"
        )
    return start_index, end_index


def verify_mapped_text(
    text_page: PdfTextPage,
    char_start: int,
    char_count: int,
    expected: str,
) -> None:
    """Verify PDFium's continuous extraction reproduces the mapped characters.

    The per-character offset map defines the boundaries, but ``get_text_range``
    may rewrite or drop characters when a whole range is requested at once
    instead of one character at a time. When the continuous extraction
    disagrees with the mapped text, the range cannot be proven to describe
    ``expected`` exactly, so mapping fails instead of returning geometry for
    text the caller did not ask for.
    """
    continuous = text_page.get_text_range(char_start, char_count)
    if continuous != expected:
        raise EvidenceMappingError(
            "PDFium continuous character range does not reproduce the mapped text"
        )


@dataclass(slots=True)
class _PageGeometry:
    """Cached PDFium objects and offset map for one page."""

    page_index: int
    page: PdfPage
    text_page: PdfTextPage
    segments: list[str]
    offsets: list[int]

    def close(self) -> None:
        """Release the PDFium page and text page."""
        self.text_page.close()
        self.page.close()


class EvidenceLocationService:
    """Locate exact page-text spans in a stored PDF without modifying state."""

    def __init__(self, document_service: DocumentService) -> None:
        self._document_service = document_service
        self._repository = document_service.repository

    @property
    def repository(self) -> DocumentRepository:
        """Return the repository shared with the document service."""
        return self._repository

    def locate(
        self,
        document_id: UUID,
        spans: Sequence[EvidenceSpan],
    ) -> list[LocatedEvidence]:
        """Return one located evidence item per requested span, in order.

        Each item is validated against the stored page text before PDFium is
        opened. A failure anywhere aborts the whole request; partial results
        are never returned. The PDF is opened exactly once and each involved
        page builds its offset map exactly once.
        """
        items = list(spans)
        self._validate_batch(items)

        try:
            pdf_path = self._document_service.get_document_path(document_id)
        except DocumentNotFoundError as exc:
            raise EvidenceDocumentNotFoundError("Document not found") from exc
        stored_pages = self._load_validated_pages(document_id, pdf_path)

        for span in items:
            self._validate_span(document_id, span, stored_pages)

        document = self._open_document(pdf_path)
        geometry: dict[int, _PageGeometry] = {}
        try:
            located: list[LocatedEvidence] = []
            for span in items:
                page_geometry = geometry.get(span.page_index)
                if page_geometry is None:
                    page_geometry = self._open_page(
                        document,
                        span.page_index,
                        stored_pages[span.page_index].text,
                    )
                    geometry[span.page_index] = page_geometry
                located.append(self._locate_one(document_id, span, stored_pages, page_geometry))
            return located
        finally:
            for page_geometry in geometry.values():
                page_geometry.close()
            document.close()

    def _validate_batch(self, items: Sequence[EvidenceSpan]) -> None:
        if not EVIDENCE_ITEMS_MIN <= len(items) <= EVIDENCE_ITEMS_MAX:
            raise EvidenceValidationError(
                f"items must contain between {EVIDENCE_ITEMS_MIN} and {EVIDENCE_ITEMS_MAX} entries"
            )
        for item in items:
            if not isinstance(item, EvidenceSpan):
                raise EvidenceValidationError("each item must be an EvidenceSpan")

    def _load_validated_pages(
        self,
        document_id: UUID,
        pdf_path: Path,
    ) -> dict[int, PageText]:
        metadata = self._repository.get_document_index_metadata(document_id)
        if metadata is None:
            raise EvidenceIndexUnavailableError("Document has not been indexed")

        if metadata.extraction_version != TEXT_EXTRACTION_VERSION:
            raise EvidenceIndexMismatchError(
                "Stored index was built with a different extraction version"
            )

        pages = self._repository.get_document_page_texts(document_id)
        if not pages:
            raise EvidenceIndexUnavailableError("Document has not been indexed")

        stored_pages = {page.page_index: page for page in pages}
        if len(stored_pages) != metadata.page_count:
            raise EvidenceIndexMismatchError("Stored index page count does not match its metadata")

        try:
            current_sha256 = hash_file(pdf_path)
        except OSError as exc:
            raise EvidenceSourceError("Stored PDF is missing or unreadable") from exc
        if current_sha256 != metadata.pdf_sha256:
            raise EvidenceIndexMismatchError("Stored PDF does not match the index")

        current_pdfium_version = detect_pdfium_version()
        if current_pdfium_version != metadata.pdfium_version:
            # A different engine build is not proof of incompatibility. Every
            # involved page is re-extracted and compared exactly below; any
            # difference raises EvidenceIndexMismatchError.
            logger.warning(
                "PDFium version %s differs from indexed version %s for document %s; "
                "falling back to strict page-text validation",
                current_pdfium_version,
                metadata.pdfium_version,
                document_id,
            )
        return stored_pages

    def _validate_span(
        self,
        document_id: UUID,
        span: EvidenceSpan,
        stored_pages: dict[int, PageText],
    ) -> None:
        if isinstance(span.page_index, bool) or not isinstance(span.page_index, int):
            raise EvidenceValidationError("page_index must be an integer")
        if isinstance(span.match_start, bool) or not isinstance(span.match_start, int):
            raise EvidenceValidationError("match_start must be an integer")
        if isinstance(span.match_end, bool) or not isinstance(span.match_end, int):
            raise EvidenceValidationError("match_end must be an integer")
        if not isinstance(span.hit_text, str):
            raise EvidenceValidationError("hit_text must be a string")

        page = stored_pages.get(span.page_index)
        if page is None:
            raise EvidenceValidationError(
                f"page_index {span.page_index} is outside the indexed document"
            )

        text = page.text
        if span.match_start < 0 or span.match_end <= span.match_start:
            raise EvidenceValidationError("span must satisfy 0 <= match_start < match_end")
        if span.match_end > len(text):
            raise EvidenceValidationError("span is outside the stored page text")
        if text[span.match_start : span.match_end] != span.hit_text:
            raise EvidenceValidationError("hit_text does not match the stored page text")

        if span.chunk_id is None:
            return

        chunk = self._repository.get_chunk(span.chunk_id)
        if chunk is None:
            raise EvidenceValidationError("chunk_id does not exist")
        if chunk.document_id != str(document_id):
            raise EvidenceValidationError("chunk_id does not belong to this document")
        if chunk.extraction_version != TEXT_EXTRACTION_VERSION:
            raise EvidenceIndexMismatchError(
                "Referenced chunk was built with a different extraction version"
            )
        if chunk.page_index != span.page_index:
            raise EvidenceValidationError("chunk_id does not belong to this page")
        if chunk.start_offset > span.match_start or span.match_end > chunk.end_offset:
            raise EvidenceValidationError("span is outside the referenced chunk")
        local_start = span.match_start - chunk.start_offset
        local_end = span.match_end - chunk.start_offset
        if chunk.text[local_start:local_end] != span.hit_text:
            raise EvidenceValidationError("hit_text does not match the referenced chunk")

    def _open_document(self, pdf_path: Path) -> PdfDocument:
        try:
            return PdfDocument(str(pdf_path))
        except PdfiumError as exc:
            raise EvidenceSourceError("Stored PDF could not be opened") from exc
        except OSError as exc:
            raise EvidenceSourceError("Stored PDF is missing or unreadable") from exc

    def _open_page(
        self,
        document: PdfDocument,
        page_index: int,
        stored_text: str,
    ) -> _PageGeometry:
        try:
            page = document[page_index]
            text_page = page.get_textpage()
        except (PdfiumError, IndexError) as exc:
            raise EvidenceSourceError("Stored PDF page could not be read") from exc

        try:
            segments, offsets = build_offset_map(text_page)
            if "".join(segments) != stored_text:
                raise EvidenceIndexMismatchError(
                    "Current page text does not match the stored index"
                )
        except Exception:
            text_page.close()
            page.close()
            raise

        return _PageGeometry(
            page_index=page_index,
            page=page,
            text_page=text_page,
            segments=segments,
            offsets=offsets,
        )

    def _locate_one(
        self,
        document_id: UUID,
        span: EvidenceSpan,
        stored_pages: dict[int, PageText],
        geometry: _PageGeometry,
    ) -> LocatedEvidence:
        stored_text = stored_pages[span.page_index].text
        char_start, char_end = map_python_span(
            geometry.offsets,
            span.match_start,
            span.match_end,
        )
        char_count = char_end - char_start

        matched_text = "".join(geometry.segments[char_start:char_end])
        if matched_text != stored_text[span.match_start : span.match_end]:
            raise EvidenceMappingError("PDFium character range does not reproduce the stored span")
        if matched_text != span.hit_text:
            raise EvidenceMappingError("PDFium character range does not reproduce hit_text")

        try:
            verify_mapped_text(geometry.text_page, char_start, char_count, matched_text)
            rects = tuple(match_rects(geometry.text_page, char_start, char_count))
        except PdfiumError as exc:
            raise EvidenceSourceError("Stored PDF geometry could not be read") from exc
        if not rects:
            raise EvidenceMappingError("No PDFium geometry is available for this span")

        return LocatedEvidence(
            document_id=document_id,
            page_index=span.page_index,
            page_number=span.page_index + 1,
            match_start=span.match_start,
            match_end=span.match_end,
            hit_text=span.hit_text,
            chunk_id=span.chunk_id,
            char_start=char_start,
            char_count=char_count,
            matched_text=matched_text,
            rects=rects,
            page=build_page_descriptor(geometry.page, span.page_index),
        )


__all__ = [
    "EVIDENCE_ITEMS_MAX",
    "EVIDENCE_ITEMS_MIN",
    "EvidenceDocumentNotFoundError",
    "EvidenceIndexMismatchError",
    "EvidenceIndexUnavailableError",
    "EvidenceLocationError",
    "EvidenceLocationService",
    "EvidenceMappingError",
    "EvidenceSourceError",
    "EvidenceSpan",
    "EvidenceValidationError",
    "LocatedEvidence",
    "build_offset_map",
    "map_python_span",
    "verify_mapped_text",
]
