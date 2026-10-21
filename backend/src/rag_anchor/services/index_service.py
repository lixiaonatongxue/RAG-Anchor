"""Index one project-managed PDF into traceable SQLite rows.

The service performs the whole index pipeline for a document that already
exists in the ``documents`` table:

1. resolve the stored PDF through :class:`DocumentService`,
2. hash the PDF bytes,
3. extract one :class:`PageText` per page,
4. split each page with the deterministic chunker,
5. verify that every chunk is an exact slice of its stored page text,
6. drop whitespace-only chunks at the persistence layer only,
7. replace the document's index atomically in one SQLite transaction.

Extraction, chunking and verification all happen *before* the transaction is
opened, so a parse or chunking failure can never remove an existing index.
The index never rewrites ``PageText.text`` and never changes the chunking
service semantics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import (
    PackageNotFoundError,
)
from importlib.metadata import (
    version as package_version,
)
from pathlib import Path
from uuid import UUID

from rag_anchor.domain.text import (
    TEXT_EXTRACTION_VERSION,
    PageText,
    TextChunk,
)
from rag_anchor.infra.database import DocumentIndexRecord, DocumentRepository
from rag_anchor.services.chunking_service import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_page_texts,
)
from rag_anchor.services.document_service import DocumentService
from rag_anchor.services.text_extraction_service import extract_page_texts

_HASH_CHUNK_SIZE = 1024 * 1024


class DocumentIndexError(RuntimeError):
    """Base class for document indexing failures."""


class DocumentIndexSourceError(DocumentIndexError):
    """Raised when the stored PDF is missing or unreadable."""


class DocumentIndexPersistenceError(DocumentIndexError):
    """Raised when the index transaction cannot be written."""


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Statistics of one completed index write."""

    document_id: UUID
    fingerprint: str
    page_count: int
    chunk_count: int
    filtered_chunk_count: int
    indexed_at: datetime


class DocumentIndexService:
    """Build, verify and persist the index of project-managed PDFs."""

    def __init__(self, document_service: DocumentService) -> None:
        self._document_service = document_service
        self._repository = document_service.repository

    @property
    def repository(self) -> DocumentRepository:
        return self._repository

    def index_document(
        self,
        document_id: UUID,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        extraction_version: str = TEXT_EXTRACTION_VERSION,
    ) -> IndexResult:
        """Rebuild and atomically persist the index of one document.

        The PDF path always comes from the document metadata row; no
        caller-supplied path is accepted. The previous index is replaced only
        after extraction, chunking and source verification have all succeeded.

        Raises
        ------
        DocumentNotFoundError
            If ``document_id`` has no metadata row.
        DocumentIndexSourceError
            If the stored PDF file is missing or is not a regular file.
        PdfTextExtractionError
            If PDFium cannot open or read the stored PDF.
        ValueError
            If chunking parameters are invalid or a chunk does not match its
            source page.
        DocumentIndexPersistenceError
            If SQLite rejects the index transaction.
        """
        pdf_path = self._document_service.get_document_path(document_id)
        _require_readable_file(pdf_path)

        try:
            pdf_sha256 = hash_file(pdf_path)
        except OSError as exc:
            raise DocumentIndexSourceError(f"Stored PDF is not readable: {pdf_path}") from exc
        pages = extract_page_texts(pdf_path, document_id=document_id)
        chunks = chunk_page_texts(
            pages,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            extraction_version=extraction_version,
        )

        pages_by_index = {page.page_index: page for page in pages}
        persisted_chunks: list[TextChunk] = []
        filtered_chunk_count = 0
        for chunk in chunks:
            page = pages_by_index.get(chunk.page_index)
            if page is None:
                raise ValueError("chunk references a page that was not extracted")
            chunk.verify_source(page)
            if not chunk.text.strip():
                # Filtering happens only here: ``PageText`` and the chunking
                # service still return the original whitespace verbatim.
                filtered_chunk_count += 1
                continue
            persisted_chunks.append(chunk)

        pdfium_version = detect_pdfium_version()
        fingerprint = compute_index_fingerprint(
            pdf_sha256=pdf_sha256,
            extraction_version=extraction_version,
            pdfium_version=pdfium_version,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        indexed_at = datetime.now(UTC)
        metadata = DocumentIndexRecord(
            document_id=document_id,
            pdf_sha256=pdf_sha256,
            extraction_version=extraction_version,
            pdfium_version=pdfium_version,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            fingerprint=fingerprint,
            page_count=len(pages),
            chunk_count=len(persisted_chunks),
            filtered_chunk_count=filtered_chunk_count,
            indexed_at=indexed_at,
        )

        try:
            self._repository.replace_document_index(
                document_id,
                pages=pages,
                chunks=persisted_chunks,
                metadata=metadata,
            )
        except sqlite3.Error as exc:
            raise DocumentIndexPersistenceError("Failed to persist document index") from exc

        return IndexResult(
            document_id=document_id,
            fingerprint=fingerprint,
            page_count=len(pages),
            chunk_count=len(persisted_chunks),
            filtered_chunk_count=filtered_chunk_count,
            indexed_at=indexed_at,
        )

    def get_document_page_texts(self, document_id: UUID) -> list[PageText]:
        """Return the stored page texts of one document, ordered by page."""
        return self._repository.get_document_page_texts(document_id)

    def get_document_chunks(self, document_id: UUID) -> list[TextChunk]:
        """Return the stored chunks of one document in stable page order."""
        return self._repository.get_document_chunks(document_id)

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        """Return one stored chunk by ID, or ``None``."""
        return self._repository.get_chunk(chunk_id)

    def get_index_metadata(self, document_id: UUID) -> DocumentIndexRecord | None:
        """Return the stored index configuration of one document."""
        return self._repository.get_document_index_metadata(document_id)


def hash_file(path: Path, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Return the lowercase SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def detect_pdfium_version() -> str:
    """Return the installed pypdfium2 distribution version."""
    try:
        return package_version("pypdfium2")
    except PackageNotFoundError:
        return "unknown"


def compute_index_fingerprint(
    *,
    pdf_sha256: str,
    extraction_version: str,
    pdfium_version: str,
    max_chars: int,
    overlap_chars: int,
) -> str:
    """Return a deterministic fingerprint for one index configuration.

    The digest covers the PDF bytes, extraction semantics, PDFium version and
    both chunking parameters. Any change to one of them changes the
    fingerprint; ``document_id`` is intentionally excluded because the
    fingerprint only describes the inputs of one document's index.
    """
    payload = json.dumps(
        [
            pdf_sha256,
            extraction_version,
            pdfium_version,
            max_chars,
            overlap_chars,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_readable_file(path: Path) -> None:
    if not path.exists():
        raise DocumentIndexSourceError(f"Stored PDF is missing: {path}")
    if not path.is_file():
        raise DocumentIndexSourceError(f"Stored PDF path is not a file: {path}")
