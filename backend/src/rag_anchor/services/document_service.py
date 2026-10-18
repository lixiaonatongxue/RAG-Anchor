"""Document upload, PDF validation and metadata management.

Uploads are streamed into a temporary file, validated by actually opening them
with PDFium, and only then moved into the documents directory. The database row
is written last; if that fails the stored PDF is removed again so an API error
does not leave an orphan file behind.
"""

from __future__ import annotations

import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile
from pypdfium2 import PdfDocument, PdfiumError

from rag_anchor.config import UPLOAD_CHUNK_SIZE, Settings
from rag_anchor.infra.database import DocumentRecord, DocumentRepository

logger = logging.getLogger(__name__)

_DEFAULT_FILENAME = "upload.pdf"
_MAX_FILENAME_LENGTH = 255


class DocumentServiceError(RuntimeError):
    """Base class for document service failures."""


class InvalidUploadError(DocumentServiceError):
    """Raised when the uploaded bytes are not a readable PDF."""


class UploadTooLargeError(DocumentServiceError):
    """Raised when an upload exceeds the configured maximum size."""


class DocumentNotFoundError(DocumentServiceError):
    """Raised when the requested document does not exist."""


class DocumentPersistenceError(DocumentServiceError):
    """Raised when the PDF file or its metadata cannot be persisted."""


class DocumentService:
    """Coordinates file storage with the SQLite metadata repository."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._repository = DocumentRepository(settings.database_path)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def repository(self) -> DocumentRepository:
        return self._repository

    def initialize(self) -> None:
        """Prepare runtime directories and the database schema."""
        self._settings.ensure_directories()
        self._repository.initialize()

    async def create_document(self, upload: UploadFile) -> DocumentRecord:
        """Store one uploaded PDF and return its metadata."""
        filename = sanitize_filename(upload.filename)
        self._settings.ensure_directories()

        file_descriptor, temp_name = tempfile.mkstemp(
            dir=self._settings.temp_dir,
            prefix="upload-",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        size_bytes = 0
        final_path: Path | None = None

        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    size_bytes += len(chunk)
                    if size_bytes > self._settings.max_upload_bytes:
                        raise UploadTooLargeError(
                            "Uploaded file exceeds the maximum size of "
                            f"{self._settings.max_upload_bytes} bytes"
                        )
                    handle.write(chunk)

            page_count = inspect_pdf(temp_path)

            document_id = uuid4()
            while True:
                candidate = self._settings.documents_dir / f"{document_id}.pdf"
                if not candidate.exists():
                    break
                document_id = uuid4()

            final_path = candidate
            try:
                os.replace(temp_path, final_path)
            except OSError as exc:
                raise DocumentPersistenceError("Failed to store uploaded PDF") from exc

            record = DocumentRecord(
                document_id=document_id,
                filename=filename,
                size_bytes=size_bytes,
                page_count=page_count,
                created_at=datetime.now(UTC),
            )
            try:
                self._repository.create(record)
            except Exception as exc:
                try:
                    final_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Failed to remove stored PDF after metadata failure")
                raise DocumentPersistenceError("Failed to persist document metadata") from exc

            return record
        finally:
            await upload.close()
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to remove temporary upload file")

    def list_documents(self, *, limit: int, offset: int) -> tuple[list[DocumentRecord], int]:
        """Return one page of records plus the total document count."""
        return self._repository.list(limit=limit, offset=offset), self._repository.count()

    def get_document(self, document_id: UUID) -> DocumentRecord | None:
        """Return one document's metadata or ``None``."""
        return self._repository.get(document_id)

    def get_document_path(self, document_id: UUID) -> Path:
        """Return the server-managed PDF path for a known document.

        The path is always derived from the UUID, never from client-supplied
        filenames. A missing metadata row raises ``DocumentNotFoundError``;
        the caller decides how to report a missing or unreadable file.
        """
        if self._repository.get(document_id) is None:
            raise DocumentNotFoundError("Document not found")
        return self._settings.documents_dir / f"{document_id}.pdf"

    def delete_document(self, document_id: UUID) -> None:
        """Delete metadata and the stored PDF.

        The database row is the source of truth. File removal is best effort so
        a missing file does not turn a successful metadata deletion into a
        failure; any remaining file is only an orphan and never exposed to
        clients.
        """
        if not self._repository.delete(document_id):
            raise DocumentNotFoundError("Document not found")

        file_path = self._settings.documents_dir / f"{document_id}.pdf"
        try:
            file_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove PDF file for document %s", document_id)


def inspect_pdf(path: Path) -> int:
    """Open ``path`` with PDFium and return its page count.

    The MIME type and filename extension are deliberately ignored: the only
    trusted signal is whether PDFium can actually parse the file.
    """
    try:
        with PdfDocument(str(path)) as document:
            page_count = len(document)
    except PdfiumError as exc:
        raise InvalidUploadError("Uploaded file is not a valid PDF") from exc
    except OSError as exc:
        raise InvalidUploadError("Uploaded PDF could not be read") from exc

    if page_count < 1:
        raise InvalidUploadError("Uploaded PDF contains no pages")
    return page_count


def sanitize_filename(raw: str | None) -> str:
    """Return a display-only basename that can never select a disk path.

    Both POSIX and Windows separators are stripped. The result is stored as
    metadata only; the actual file name is always the generated document ID.
    """
    if not raw:
        return _DEFAULT_FILENAME

    name = raw.replace("\\", "/").rsplit("/", 1)[-1]
    name = name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        return _DEFAULT_FILENAME
    return name[:_MAX_FILENAME_LENGTH]
