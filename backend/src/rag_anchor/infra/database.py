"""SQLite persistence for document metadata.

The standard library ``sqlite3`` module is used directly; the MVP does not need
an ORM. Every operation opens a short-lived connection so request handlers can
run in FastAPI's worker threads without sharing a connection object.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

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
"""


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Metadata stored for one uploaded PDF."""

    document_id: UUID
    filename: str
    size_bytes: int
    page_count: int
    created_at: datetime


class DocumentRepository:
    """SQLite-backed metadata repository."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._database_path)
        connection.row_factory = sqlite3.Row
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
        """Delete one row, returning whether a record was removed."""
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE document_id = ?",
                (str(document_id),),
            )
            return cursor.rowcount > 0


def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        document_id=UUID(row["document_id"]),
        filename=row["filename"],
        size_bytes=int(row["size_bytes"]),
        page_count=int(row["page_count"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
