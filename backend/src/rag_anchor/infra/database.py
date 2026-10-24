"""SQLite persistence for document metadata.

The standard library ``sqlite3`` module is used directly; the MVP does not need
an ORM. Every operation opens a short-lived connection so request handlers can
run in FastAPI's worker threads without sharing a connection object.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from rag_anchor.domain.text import PageText, TextChunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    page_count INTEGER NOT NULL CHECK (page_count >= 1),
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON documents (created_at DESC, document_id ASC);

CREATE TABLE IF NOT EXISTS document_page_texts (
    document_id TEXT NOT NULL,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    page_number INTEGER NOT NULL CHECK (page_number = page_index + 1),
    text TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    PRIMARY KEY (document_id, page_index),
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    page_number INTEGER NOT NULL CHECK (page_number = page_index + 1),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    text TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    UNIQUE (document_id, page_index, chunk_index),
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_page
    ON document_chunks (document_id, page_index, chunk_index);

CREATE TABLE IF NOT EXISTS document_index_metadata (
    document_id TEXT PRIMARY KEY,
    pdf_sha256 TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    pdfium_version TEXT NOT NULL,
    max_chars INTEGER NOT NULL CHECK (max_chars > 0),
    overlap_chars INTEGER NOT NULL CHECK (overlap_chars >= 0),
    fingerprint TEXT NOT NULL,
    page_count INTEGER NOT NULL CHECK (page_count >= 0),
    chunk_count INTEGER NOT NULL CHECK (chunk_count >= 0),
    filtered_chunk_count INTEGER NOT NULL CHECK (filtered_chunk_count >= 0),
    indexed_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semantic_index_metadata (
    document_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_digest TEXT NOT NULL DEFAULT '',
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    normalization TEXT NOT NULL,
    index_version INTEGER NOT NULL DEFAULT 0,
    candidate_version INTEGER NOT NULL DEFAULT 0,
    source_fingerprint TEXT NOT NULL,
    source_extraction_version TEXT NOT NULL,
    candidate_count INTEGER NOT NULL CHECK (candidate_count >= 0),
    built_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS semantic_candidates (
    candidate_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    page_number INTEGER NOT NULL CHECK (page_number = page_index + 1),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
    source_text TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    vector BLOB NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents (document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_semantic_candidates_document_page
    ON semantic_candidates (document_id, page_index, start_offset);
"""

# ``CREATE TABLE IF NOT EXISTS`` cannot add columns to a database created by an
# older version. These additive migrations keep the source text and the derived
# rows intact; old metadata rows keep the conservative defaults (version 0 and
# an empty digest), so they are reported as stale until rebuilt.
_SEMANTIC_METADATA_MIGRATIONS = (
    ("model_digest", "model_digest TEXT NOT NULL DEFAULT ''"),
    ("index_version", "index_version INTEGER NOT NULL DEFAULT 0"),
    ("candidate_version", "candidate_version INTEGER NOT NULL DEFAULT 0"),
)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Metadata stored for one uploaded PDF."""

    document_id: UUID
    filename: str
    size_bytes: int
    page_count: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentIndexRecord:
    """Configuration and statistics of one completed document index."""

    document_id: UUID
    pdf_sha256: str
    extraction_version: str
    pdfium_version: str
    max_chars: int
    overlap_chars: int
    fingerprint: str
    page_count: int
    chunk_count: int
    filtered_chunk_count: int
    indexed_at: datetime


@dataclass(frozen=True, slots=True)
class SemanticIndexRecord:
    """Metadata of one derived semantic index.

    ``source_fingerprint`` ties the vectors to the text index they were built
    from. ``model_name``, ``model_version``, ``dimension`` and ``normalization``
    describe how the vectors were produced and must match the active provider
    before the index may be queried.
    """

    document_id: UUID
    model_name: str
    model_version: str
    dimension: int
    normalization: str
    source_fingerprint: str
    source_extraction_version: str
    candidate_count: int
    built_at: datetime
    model_digest: str = ""
    index_version: int = 0
    candidate_version: int = 0


@dataclass(frozen=True, slots=True)
class SemanticCandidateRecord:
    """One persisted semantic candidate and its encoded vector."""

    candidate_id: str
    document_id: UUID
    chunk_id: str
    page_index: int
    page_number: int
    chunk_index: int
    start_offset: int
    end_offset: int
    source_text: str
    extraction_version: str
    vector: bytes


class DocumentRepository:
    """SQLite-backed metadata repository."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the database file and schema if they are missing."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)
            self._migrate_semantic_metadata(connection)

    @staticmethod
    def _migrate_semantic_metadata(connection: sqlite3.Connection) -> None:
        """Add version/identity columns to databases created by older builds."""
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(semantic_index_metadata)")
        }
        for name, definition in _SEMANTIC_METADATA_MIGRATIONS:
            if name not in columns:
                connection.execute(f"ALTER TABLE semantic_index_metadata ADD COLUMN {definition}")

    def create(self, record: DocumentRecord) -> None:
        """Insert one metadata row."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (document_id, filename, size_bytes, page_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(record.document_id),
                    record.filename,
                    record.size_bytes,
                    record.page_count,
                    record.created_at.isoformat(),
                ),
            )

    def get(self, document_id: UUID) -> DocumentRecord | None:
        """Return one record or ``None`` when the ID is unknown."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, filename, size_bytes, page_count, created_at
                FROM documents
                WHERE document_id = ?
                """,
                (str(document_id),),
            ).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(self, *, limit: int, offset: int) -> list[DocumentRecord]:
        """Return one page of records, newest first with a stable tie-breaker."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, filename, size_bytes, page_count, created_at
                FROM documents
                ORDER BY created_at DESC, document_id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    def count(self) -> int:
        """Return the total number of stored documents."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM documents").fetchone()
        return int(row["total"])

    def delete(self, document_id: UUID) -> bool:
        """Delete one row, returning whether a record was removed.

        Child index rows are removed by ``ON DELETE CASCADE`` because
        ``_connect()`` enables foreign keys for every connection.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (str(document_id),),
            )
            return cursor.rowcount > 0

    def replace_document_index(
        self,
        document_id: UUID,
        *,
        pages: Sequence[PageText],
        chunks: Sequence[TextChunk],
        metadata: DocumentIndexRecord,
    ) -> None:
        """Atomically replace every stored index row for one document.

        Extraction, chunking and source verification happen outside this
        method. Only after all inputs exist is one connection opened: the old
        page texts, chunks and metadata are deleted and the new rows inserted
        inside a single transaction. Any failure rolls the whole replacement
        back, so a failed re-index can never leave a half-written index or a
        mixture of two index versions.

        Raises
        ------
        ValueError
            If ``metadata`` or any page/chunk belongs to another document.
        KeyError
            If the document metadata row does not exist.
        """
        document_key = str(document_id)
        if metadata.document_id != document_id:
            raise ValueError("metadata document_id does not match the target document")
        for page in pages:
            if page.document_id != document_key:
                raise ValueError("page text does not belong to the target document")
        for chunk in chunks:
            if chunk.document_id != document_key:
                raise ValueError("chunk does not belong to the target document")

        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?",
                (document_key,),
            ).fetchone()
            if exists is None:
                raise KeyError(document_key)

            connection.execute(
                "DELETE FROM semantic_candidates WHERE document_id = ?",
                (document_key,),
            )
            connection.execute(
                "DELETE FROM semantic_index_metadata WHERE document_id = ?",
                (document_key,),
            )
            connection.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document_key,),
            )
            connection.execute(
                "DELETE FROM document_page_texts WHERE document_id = ?",
                (document_key,),
            )
            connection.execute(
                "DELETE FROM document_index_metadata WHERE document_id = ?",
                (document_key,),
            )

            connection.executemany(
                """
                INSERT INTO document_page_texts
                    (document_id, page_index, page_number, text, extraction_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document_key,
                        page.page_index,
                        page.page_number,
                        page.text,
                        metadata.extraction_version,
                    )
                    for page in pages
                ],
            )
            connection.executemany(
                """
                INSERT INTO document_chunks
                    (chunk_id, document_id, page_index, page_number, chunk_index,
                     start_offset, end_offset, text, extraction_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        document_key,
                        chunk.page_index,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.text,
                        chunk.extraction_version,
                    )
                    for chunk in chunks
                ],
            )
            connection.execute(
                """
                INSERT INTO document_index_metadata
                    (document_id, pdf_sha256, extraction_version, pdfium_version,
                     max_chars, overlap_chars, fingerprint, page_count,
                     chunk_count, filtered_chunk_count, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_key,
                    metadata.pdf_sha256,
                    metadata.extraction_version,
                    metadata.pdfium_version,
                    metadata.max_chars,
                    metadata.overlap_chars,
                    metadata.fingerprint,
                    metadata.page_count,
                    metadata.chunk_count,
                    metadata.filtered_chunk_count,
                    metadata.indexed_at.isoformat(),
                ),
            )

    def get_document_page_texts(self, document_id: UUID) -> list[PageText]:
        """Return stored page texts ordered by ``page_index``."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, page_index, page_number, text
                FROM document_page_texts
                WHERE document_id = ?
                ORDER BY page_index ASC
                """,
                (str(document_id),),
            ).fetchall()
        return [_row_to_page_text(row) for row in rows]

    def get_document_chunks(self, document_id: UUID) -> list[TextChunk]:
        """Return stored chunks ordered by page then ``chunk_index``."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, page_index, page_number,
                       chunk_index, start_offset, end_offset, text,
                       extraction_version
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY page_index ASC, chunk_index ASC
                """,
                (str(document_id),),
            ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def list_chunks(self, *, document_id: UUID | None = None) -> list[TextChunk]:
        """Return persisted chunks in deterministic retrieval order.

        Retrieval never re-parses a PDF: this method is the only source of
        chunk text. Ordering is explicit so callers can reproduce results even
        if SQLite changes its physical row order.
        """
        with self._connect() as connection:
            if document_id is None:
                rows = connection.execute(
                    """
                    SELECT chunk_id, document_id, page_index, page_number,
                           chunk_index, start_offset, end_offset, text,
                           extraction_version
                    FROM document_chunks
                    ORDER BY document_id ASC, page_index ASC, chunk_index ASC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT chunk_id, document_id, page_index, page_number,
                           chunk_index, start_offset, end_offset, text,
                           extraction_version
                    FROM document_chunks
                    WHERE document_id = ?
                    ORDER BY page_index ASC, chunk_index ASC
                    """,
                    (str(document_id),),
                ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        """Return one chunk by ID, or ``None`` when it is unknown."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT chunk_id, document_id, page_index, page_number,
                       chunk_index, start_offset, end_offset, text,
                       extraction_version
                FROM document_chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        return _row_to_chunk(row) if row is not None else None

    def get_document_index_metadata(self, document_id: UUID) -> DocumentIndexRecord | None:
        """Return the stored index configuration for one document."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, pdf_sha256, extraction_version, pdfium_version,
                       max_chars, overlap_chars, fingerprint, page_count,
                       chunk_count, filtered_chunk_count, indexed_at
                FROM document_index_metadata
                WHERE document_id = ?
                """,
                (str(document_id),),
            ).fetchone()
        return _row_to_index_record(row) if row is not None else None

    def delete_document_index(self, document_id: UUID) -> bool:
        """Remove all index rows for one document without deleting the document.

        Normally unnecessary because child tables cascade from ``documents``;
        this exists for explicit cleanup and for repair scenarios.
        """
        document_key = str(document_id)
        with self._connect() as connection:
            deleted_chunks = connection.execute(
                "DELETE FROM document_chunks WHERE document_id = ?",
                (document_key,),
            ).rowcount
            deleted_pages = connection.execute(
                "DELETE FROM document_page_texts WHERE document_id = ?",
                (document_key,),
            ).rowcount
            deleted_metadata = connection.execute(
                "DELETE FROM document_index_metadata WHERE document_id = ?",
                (document_key,),
            ).rowcount
            deleted_semantic_candidates = connection.execute(
                "DELETE FROM semantic_candidates WHERE document_id = ?",
                (document_key,),
            ).rowcount
            deleted_semantic_metadata = connection.execute(
                "DELETE FROM semantic_index_metadata WHERE document_id = ?",
                (document_key,),
            ).rowcount
        return bool(
            deleted_chunks
            or deleted_pages
            or deleted_metadata
            or deleted_semantic_candidates
            or deleted_semantic_metadata
        )

    def replace_semantic_index(
        self,
        document_id: UUID,
        *,
        metadata: SemanticIndexRecord,
        candidates: Sequence[SemanticCandidateRecord],
    ) -> None:
        """Atomically replace every derived semantic row of one document.

        The text index is never touched. Vectors are persisted only after the
        caller has embedded and validated every candidate; one transaction
        removes the previous semantic index and inserts the new one. A failure
        leaves the previous derived index (or its absence) unchanged.

        Raises
        ------
        ValueError
            If ``metadata`` or any candidate belongs to another document.
        KeyError
            If the document metadata row does not exist.
        """
        document_key = str(document_id)
        if metadata.document_id != document_id:
            raise ValueError("semantic metadata document_id does not match the target")
        for candidate in candidates:
            if candidate.document_id != document_id:
                raise ValueError("semantic candidate does not belong to the target document")

        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE document_id = ?",
                (document_key,),
            ).fetchone()
            if exists is None:
                raise KeyError(document_key)

            connection.execute(
                "DELETE FROM semantic_candidates WHERE document_id = ?",
                (document_key,),
            )
            connection.execute(
                "DELETE FROM semantic_index_metadata WHERE document_id = ?",
                (document_key,),
            )
            connection.executemany(
                """
                INSERT INTO semantic_candidates
                    (candidate_id, document_id, chunk_id, page_index, page_number,
                     chunk_index, start_offset, end_offset, source_text,
                     extraction_version, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        candidate.candidate_id,
                        document_key,
                        candidate.chunk_id,
                        candidate.page_index,
                        candidate.page_number,
                        candidate.chunk_index,
                        candidate.start_offset,
                        candidate.end_offset,
                        candidate.source_text,
                        candidate.extraction_version,
                        candidate.vector,
                    )
                    for candidate in candidates
                ],
            )
            connection.execute(
                """
                INSERT INTO semantic_index_metadata
                    (document_id, model_name, model_version, model_digest, dimension,
                     normalization, index_version, candidate_version,
                     source_fingerprint, source_extraction_version,
                     candidate_count, built_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_key,
                    metadata.model_name,
                    metadata.model_version,
                    metadata.model_digest,
                    metadata.dimension,
                    metadata.normalization,
                    metadata.index_version,
                    metadata.candidate_version,
                    metadata.source_fingerprint,
                    metadata.source_extraction_version,
                    metadata.candidate_count,
                    metadata.built_at.isoformat(),
                ),
            )

    def get_semantic_index_metadata(self, document_id: UUID) -> SemanticIndexRecord | None:
        """Return the derived semantic index metadata, if it exists."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, model_name, model_version, model_digest,
                       dimension, normalization, index_version, candidate_version,
                       source_fingerprint, source_extraction_version,
                       candidate_count, built_at
                FROM semantic_index_metadata
                WHERE document_id = ?
                """,
                (str(document_id),),
            ).fetchone()
        return _row_to_semantic_index(row) if row is not None else None

    def list_semantic_candidates(self, document_id: UUID) -> list[SemanticCandidateRecord]:
        """Return every persisted candidate of one document in stable order."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id, document_id, chunk_id, page_index, page_number,
                       chunk_index, start_offset, end_offset, source_text,
                       extraction_version, vector
                FROM semantic_candidates
                WHERE document_id = ?
                ORDER BY page_index ASC, start_offset ASC, chunk_index ASC, candidate_id ASC
                """,
                (str(document_id),),
            ).fetchall()
        return [_row_to_semantic_candidate(row) for row in rows]

    def delete_semantic_index(self, document_id: UUID) -> bool:
        """Remove the derived semantic index of one document, keeping its text index."""
        document_key = str(document_id)
        with self._connect() as connection:
            deleted_candidates = connection.execute(
                "DELETE FROM semantic_candidates WHERE document_id = ?",
                (document_key,),
            ).rowcount
            deleted_metadata = connection.execute(
                "DELETE FROM semantic_index_metadata WHERE document_id = ?",
                (document_key,),
            ).rowcount
        return bool(deleted_candidates or deleted_metadata)


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        document_id=UUID(row["document_id"]),
        filename=row["filename"],
        size_bytes=int(row["size_bytes"]),
        page_count=int(row["page_count"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_page_text(row: sqlite3.Row) -> PageText:
    return PageText(
        document_id=row["document_id"],
        page_index=int(row["page_index"]),
        page_number=int(row["page_number"]),
        text=row["text"],
    )


def _row_to_chunk(row: sqlite3.Row) -> TextChunk:
    return TextChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        page_index=int(row["page_index"]),
        page_number=int(row["page_number"]),
        start_offset=int(row["start_offset"]),
        end_offset=int(row["end_offset"]),
        text=row["text"],
        chunk_index=int(row["chunk_index"]),
        extraction_version=row["extraction_version"],
    )


def _row_to_index_record(row: sqlite3.Row) -> DocumentIndexRecord:
    return DocumentIndexRecord(
        document_id=UUID(row["document_id"]),
        pdf_sha256=row["pdf_sha256"],
        extraction_version=row["extraction_version"],
        pdfium_version=row["pdfium_version"],
        max_chars=int(row["max_chars"]),
        overlap_chars=int(row["overlap_chars"]),
        fingerprint=row["fingerprint"],
        page_count=int(row["page_count"]),
        chunk_count=int(row["chunk_count"]),
        filtered_chunk_count=int(row["filtered_chunk_count"]),
        indexed_at=datetime.fromisoformat(row["indexed_at"]),
    )


def _row_to_semantic_index(row: sqlite3.Row) -> SemanticIndexRecord:
    return SemanticIndexRecord(
        document_id=UUID(row["document_id"]),
        model_name=row["model_name"],
        model_version=row["model_version"],
        model_digest=row["model_digest"],
        dimension=int(row["dimension"]),
        normalization=row["normalization"],
        index_version=int(row["index_version"]),
        candidate_version=int(row["candidate_version"]),
        source_fingerprint=row["source_fingerprint"],
        source_extraction_version=row["source_extraction_version"],
        candidate_count=int(row["candidate_count"]),
        built_at=datetime.fromisoformat(row["built_at"]),
    )


def _row_to_semantic_candidate(row: sqlite3.Row) -> SemanticCandidateRecord:
    return SemanticCandidateRecord(
        candidate_id=row["candidate_id"],
        document_id=UUID(row["document_id"]),
        chunk_id=row["chunk_id"],
        page_index=int(row["page_index"]),
        page_number=int(row["page_number"]),
        chunk_index=int(row["chunk_index"]),
        start_offset=int(row["start_offset"]),
        end_offset=int(row["end_offset"]),
        source_text=row["source_text"],
        extraction_version=row["extraction_version"],
        vector=bytes(row["vector"]),
    )
