"""M1-01B tests: SQLite persistence of traceable chunks and page text.

Every test uses a temporary data directory, a temporary SQLite database and
PDFs generated at runtime. The user's real ``backend/.data`` store is never
touched.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from rag_anchor.config import Settings
from rag_anchor.domain import (
    TEXT_EXTRACTION_VERSION,
    build_chunk_id,
)
from rag_anchor.infra import (
    DocumentRecord,
    DocumentRepository,
)
from rag_anchor.services import (
    DocumentIndexPersistenceError,
    DocumentIndexService,
    DocumentIndexSourceError,
    DocumentNotFoundError,
    DocumentService,
    PdfTextOpenError,
    chunk_page_texts,
    extract_page_texts,
    search_anchors,
)

DOCUMENTS_URL = "/api/v1/documents"

_COUNT_SQL = {
    "documents": "SELECT COUNT(*) FROM documents",
    "page_texts": "SELECT COUNT(*) FROM document_page_texts",
    "chunks": "SELECT COUNT(*) FROM document_chunks",
    "metadata": "SELECT COUNT(*) FROM document_index_metadata",
}

_INDEX_TABLES = {
    "document_page_texts",
    "document_chunks",
    "document_index_metadata",
}


def _prepare_service(tmp_path: Path, name: str = "data") -> tuple[Settings, DocumentService]:
    """Create an isolated settings object and initialized document service."""
    settings = Settings(data_dir=tmp_path / name)
    service = DocumentService(settings)
    service.initialize()
    return settings, service


def _store_document(
    service: DocumentService,
    settings: Settings,
    source: Path,
    *,
    page_count: int,
) -> UUID:
    """Store ``source`` exactly the way the upload flow stores a PDF.

    The file is named after the generated document id and a matching
    ``documents`` row is inserted, so the index service resolves the PDF only
    through the project-managed path.
    """
    document_id = uuid4()
    target = settings.documents_dir / f"{document_id}.pdf"
    shutil.copyfile(source, target)
    service.repository.create(
        DocumentRecord(
            document_id=document_id,
            filename=source.name,
            size_bytes=target.stat().st_size,
            page_count=page_count,
            created_at=datetime.now(UTC),
        )
    )
    return document_id


def _table_names(database_path: Path) -> set[str]:
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row[0] for row in rows}


def _index_names(database_path: Path) -> set[str]:
    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {row[0] for row in rows}


def _row_count(database_path: Path, table: str) -> int:
    with closing(sqlite3.connect(database_path)) as connection:
        value = connection.execute(_COUNT_SQL[table]).fetchone()
    return int(value[0])


def test_fresh_database_initialization_creates_index_schema(tmp_path: Path) -> None:
    """Category 1: a new database contains every index table and index."""
    settings, service = _prepare_service(tmp_path)

    assert settings.database_path.is_file()
    assert _table_names(settings.database_path) >= _INDEX_TABLES
    assert {
        "idx_documents_created_at",
        "idx_document_chunks_document_page",
    } <= _index_names(settings.database_path)

    with service.repository._connect() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_legacy_database_is_upgraded_without_losing_documents(tmp_path: Path) -> None:
    """Category 2: an old database with only ``documents`` is upgraded in place."""
    data_dir = tmp_path / "legacy"
    data_dir.mkdir()
    database_path = data_dir / "metadata.sqlite3"
    document_id = uuid4()

    with closing(sqlite3.connect(database_path)) as connection:
        connection.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                page_count INTEGER NOT NULL CHECK (page_count >= 1),
                created_at TEXT NOT NULL
            );

            CREATE INDEX idx_documents_created_at
                ON documents (created_at DESC, document_id ASC);
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            (str(document_id), "legacy.pdf", 10, 1, datetime.now(UTC).isoformat()),
        )
        connection.commit()

    repository = DocumentRepository(database_path)
    repository.initialize()

    record = repository.get(document_id)
    assert record is not None
    assert record.filename == "legacy.pdf"
    assert _table_names(database_path) >= _INDEX_TABLES


def test_single_page_document_is_indexed(tmp_path: Path, single_page_pdf: Path) -> None:
    """Category 3: a one-page PDF yields page text plus chunks."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)

    result = indexer.index_document(document_id, max_chars=20)

    assert result.document_id == document_id
    assert result.page_count == 1
    assert result.chunk_count > 0
    assert result.filtered_chunk_count == 0
    pages = indexer.get_document_page_texts(document_id)
    assert [page.page_index for page in pages] == [0]
    chunks = indexer.get_document_chunks(document_id)
    assert chunks
    assert {chunk.page_index for chunk in chunks} == {0}


def test_multi_page_document_keeps_stable_page_order(tmp_path: Path, multi_page_pdf: Path) -> None:
    """Category 4: pages and chunks stay ordered by page and chunk index."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, multi_page_pdf, page_count=3)
    indexer = DocumentIndexService(service)

    indexer.index_document(document_id, max_chars=10)

    pages = indexer.get_document_page_texts(document_id)
    assert [page.page_index for page in pages] == [0, 1, 2]
    assert [page.page_number for page in pages] == [1, 2, 3]
    chunks = indexer.get_document_chunks(document_id)
    ordering = [(chunk.page_index, chunk.chunk_index) for chunk in chunks]
    assert ordering == sorted(ordering)
    assert {chunk.page_index for chunk in chunks} == {0, 1, 2}


def test_empty_page_text_is_persisted_without_chunks(tmp_path: Path, pdf_factory) -> None:
    """Category 5: a truly empty page keeps its PageText and emits no chunk."""
    settings, service = _prepare_service(tmp_path)
    pdf_path = pdf_factory("blank.pdf", [[]])
    document_id = _store_document(service, settings, pdf_path, page_count=1)
    indexer = DocumentIndexService(service)

    result = indexer.index_document(document_id, max_chars=20)

    pages = indexer.get_document_page_texts(document_id)
    assert len(pages) == 1
    assert pages[0].text == ""
    assert indexer.get_document_chunks(document_id) == []
    assert result.chunk_count == 0
    assert result.filtered_chunk_count == 0


def test_whitespace_only_page_is_stored_but_not_indexed(tmp_path: Path, pdf_factory) -> None:
    """Category 6: whitespace-only text is stored verbatim and filtered from chunks."""
    settings, service = _prepare_service(tmp_path)
    pdf_path = pdf_factory("whitespace.pdf", [[(100.0, 700.0, "   ")]])
    document_id = _store_document(service, settings, pdf_path, page_count=1)
    indexer = DocumentIndexService(service)

    result = indexer.index_document(document_id, max_chars=2)

    pages = indexer.get_document_page_texts(document_id)
    assert len(pages) == 1
    assert pages[0].text != ""
    assert pages[0].text.strip() == ""
    assert result.chunk_count == 0
    assert result.filtered_chunk_count == 1
    assert indexer.get_document_chunks(document_id) == []


def test_chunk_fields_round_trip_exactly(tmp_path: Path, single_page_pdf: Path) -> None:
    """Category 7: every persisted chunk field matches the source chunk."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=12, overlap_chars=2)

    pdf_path = settings.documents_dir / f"{document_id}.pdf"
    pages = extract_page_texts(pdf_path, document_id=document_id)
    expected = [
        chunk
        for chunk in chunk_page_texts(pages, max_chars=12, overlap_chars=2)
        if chunk.text.strip()
    ]
    stored = indexer.get_document_chunks(document_id)

    assert [chunk.chunk_id for chunk in stored] == [chunk.chunk_id for chunk in expected]
    for stored_chunk, expected_chunk in zip(stored, expected, strict=True):
        assert stored_chunk.document_id == str(document_id)
        assert stored_chunk.page_index == expected_chunk.page_index
        assert stored_chunk.page_number == expected_chunk.page_number
        assert stored_chunk.chunk_index == expected_chunk.chunk_index
        assert stored_chunk.start_offset == expected_chunk.start_offset
        assert stored_chunk.end_offset == expected_chunk.end_offset
        assert stored_chunk.text == expected_chunk.text
        assert stored_chunk.extraction_version == TEXT_EXTRACTION_VERSION
        assert stored_chunk.chunk_id == build_chunk_id(
            document_id=str(document_id),
            page_index=expected_chunk.page_index,
            start_offset=expected_chunk.start_offset,
            end_offset=expected_chunk.end_offset,
            extraction_version=TEXT_EXTRACTION_VERSION,
        )


def test_index_is_readable_after_reopening_the_database(
    tmp_path: Path, multi_page_pdf: Path
) -> None:
    """Category 8: a second service instance reads the same persisted index."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, multi_page_pdf, page_count=3)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)
    pages_before = indexer.get_document_page_texts(document_id)
    chunks_before = indexer.get_document_chunks(document_id)
    metadata_before = indexer.get_index_metadata(document_id)

    reopened_service = DocumentService(settings)
    reopened_indexer = DocumentIndexService(reopened_service)

    assert reopened_indexer.get_document_page_texts(document_id) == pages_before
    assert reopened_indexer.get_document_chunks(document_id) == chunks_before
    assert reopened_indexer.get_index_metadata(document_id) == metadata_before


def test_all_persisted_chunks_pass_source_verification(
    tmp_path: Path, multi_page_pdf: Path
) -> None:
    """Category 9: every stored chunk is an exact slice of its stored page text."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, multi_page_pdf, page_count=3)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=7, overlap_chars=2)

    pages_by_index = {
        page.page_index: page for page in indexer.get_document_page_texts(document_id)
    }
    chunks = indexer.get_document_chunks(document_id)
    assert chunks
    for chunk in chunks:
        page = pages_by_index[chunk.page_index]
        chunk.verify_source(page)
        assert page.text[chunk.start_offset : chunk.end_offset] == chunk.text


def test_reindex_with_same_configuration_is_idempotent(
    tmp_path: Path, multi_page_pdf: Path
) -> None:
    """Category 10: re-indexing with identical inputs is a no-op."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, multi_page_pdf, page_count=3)
    indexer = DocumentIndexService(service)

    first = indexer.index_document(document_id, max_chars=15)
    pages_after_first = indexer.get_document_page_texts(document_id)
    chunks_after_first = indexer.get_document_chunks(document_id)

    second = indexer.index_document(document_id, max_chars=15)

    assert second.fingerprint == first.fingerprint
    assert second.chunk_count == first.chunk_count
    assert indexer.get_document_page_texts(document_id) == pages_after_first
    assert indexer.get_document_chunks(document_id) == chunks_after_first
    chunk_ids = [chunk.chunk_id for chunk in chunks_after_first]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert _row_count(settings.database_path, "chunks") == len(chunk_ids)


def test_reindex_with_different_parameters_rebuilds_the_index(
    tmp_path: Path, single_page_pdf: Path
) -> None:
    """Category 11: a chunk parameter change produces a new chunk set."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)

    first = indexer.index_document(document_id, max_chars=8)
    old_ids = {chunk.chunk_id for chunk in indexer.get_document_chunks(document_id)}

    second = indexer.index_document(document_id, max_chars=64)
    new_chunks = indexer.get_document_chunks(document_id)
    new_ids = {chunk.chunk_id for chunk in new_chunks}

    assert second.fingerprint != first.fingerprint
    assert len(new_chunks) == second.chunk_count
    assert new_ids.isdisjoint(old_ids)
    assert _row_count(settings.database_path, "chunks") == len(new_chunks)


def test_pdf_content_change_triggers_a_rebuild(tmp_path: Path, pdf_factory) -> None:
    """Category 12: replacing the stored PDF updates text, chunks and fingerprint."""
    settings, service = _prepare_service(tmp_path)
    original_pdf = pdf_factory("v1.pdf", [[(100.0, 700.0, "alpha beta gamma")]])
    document_id = _store_document(service, settings, original_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    first = indexer.index_document(document_id, max_chars=100)
    old_text = indexer.get_document_page_texts(document_id)[0].text

    replacement_pdf = pdf_factory("v2.pdf", [[(100.0, 700.0, "delta epsilon zeta")]])
    shutil.copyfile(
        replacement_pdf,
        settings.documents_dir / f"{document_id}.pdf",
    )
    second = indexer.index_document(document_id, max_chars=100)

    new_text = indexer.get_document_page_texts(document_id)[0].text
    assert second.fingerprint != first.fingerprint
    assert new_text != old_text
    assert "delta" in new_text
    assert "alpha" not in new_text


def test_same_chunk_id_with_changed_text_is_overwritten(tmp_path: Path, pdf_factory) -> None:
    """Category 13: equal chunk IDs must not suppress changed text."""
    settings, service = _prepare_service(tmp_path)
    original_pdf = pdf_factory("v1.pdf", [[(100.0, 700.0, "abcdefghij")]])
    document_id = _store_document(service, settings, original_pdf, page_count=1)
    indexer = DocumentIndexService(service)

    indexer.index_document(document_id, max_chars=4)
    old_page = indexer.get_document_page_texts(document_id)[0]
    old_chunks = indexer.get_document_chunks(document_id)
    assert old_page.text.startswith("abcdefghij")
    assert old_chunks[0].text == "abcd"
    old_ids = {chunk.chunk_id for chunk in old_chunks}

    replacement_pdf = pdf_factory("v2.pdf", [[(100.0, 700.0, "ABCDEFGHIJ")]])
    shutil.copyfile(
        replacement_pdf,
        settings.documents_dir / f"{document_id}.pdf",
    )
    indexer.index_document(document_id, max_chars=4)

    new_page = indexer.get_document_page_texts(document_id)[0]
    new_chunks = indexer.get_document_chunks(document_id)

    # Same offsets and extraction version => same chunk IDs, new text.
    assert {chunk.chunk_id for chunk in new_chunks} == old_ids
    assert new_page.text.startswith("ABCDEFGHIJ")
    assert new_chunks[0].text == "ABCD"
    for chunk in new_chunks:
        assert chunk.text == new_page.text[chunk.start_offset : chunk.end_offset]
    assert _row_count(settings.database_path, "chunks") == len(new_chunks)


def test_shrinking_chunk_set_removes_stale_rows(tmp_path: Path, single_page_pdf: Path) -> None:
    """Category 14: chunks that disappeared are physically deleted."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)

    indexer.index_document(document_id, max_chars=5)
    old_chunks = indexer.get_document_chunks(document_id)
    old_ids = {chunk.chunk_id for chunk in old_chunks}

    second = indexer.index_document(document_id, max_chars=1000)
    new_chunks = indexer.get_document_chunks(document_id)

    assert second.chunk_count < len(old_chunks)
    assert {chunk.chunk_id for chunk in new_chunks}.isdisjoint(old_ids)
    assert _row_count(settings.database_path, "chunks") == len(new_chunks)


def test_parse_failure_keeps_the_previous_index_intact(
    tmp_path: Path, single_page_pdf: Path
) -> None:
    """Category 15: a corrupt PDF must not remove the existing index."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)

    pages_before = indexer.get_document_page_texts(document_id)
    chunks_before = indexer.get_document_chunks(document_id)
    metadata_before = indexer.get_index_metadata(document_id)
    assert metadata_before is not None

    (settings.documents_dir / f"{document_id}.pdf").write_bytes(b"not a pdf at all")

    with pytest.raises(PdfTextOpenError):
        indexer.index_document(document_id, max_chars=10)

    assert indexer.get_document_page_texts(document_id) == pages_before
    assert indexer.get_document_chunks(document_id) == chunks_before
    assert indexer.get_index_metadata(document_id) == metadata_before


def test_database_write_failure_rolls_back_the_transaction(
    tmp_path: Path, single_page_pdf: Path
) -> None:
    """Category 16: a constraint failure restores the previous index rows."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)

    pages_before = indexer.get_document_page_texts(document_id)
    chunks_before = indexer.get_document_chunks(document_id)
    metadata_before = indexer.get_index_metadata(document_id)
    assert metadata_before is not None
    assert chunks_before

    tampered_pages = [replace(page, text="tampered") for page in pages_before]
    broken_metadata = replace(metadata_before, page_count=-1)

    with pytest.raises(sqlite3.IntegrityError):
        service.repository.replace_document_index(
            document_id,
            pages=tampered_pages,
            chunks=[],
            metadata=broken_metadata,
        )

    # The DELETE that ran before the failing INSERT must have been rolled back.
    assert indexer.get_document_page_texts(document_id) == pages_before
    assert indexer.get_document_chunks(document_id) == chunks_before
    assert indexer.get_index_metadata(document_id) == metadata_before


def test_sqlite_error_is_translated_to_persistence_error(
    tmp_path: Path, single_page_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Category 16b: the service wraps SQLite failures in its own error type."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)
    pages_before = indexer.get_document_page_texts(document_id)

    def fail(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr(service.repository, "replace_document_index", fail)

    with pytest.raises(DocumentIndexPersistenceError):
        indexer.index_document(document_id, max_chars=10)
    assert indexer.get_document_page_texts(document_id) == pages_before


def test_missing_document_fails_without_index_rows(tmp_path: Path) -> None:
    """Category 17: an unknown document ID writes nothing."""
    settings, service = _prepare_service(tmp_path)
    indexer = DocumentIndexService(service)

    with pytest.raises(DocumentNotFoundError):
        indexer.index_document(uuid4())

    assert _row_count(settings.database_path, "page_texts") == 0
    assert _row_count(settings.database_path, "chunks") == 0
    assert _row_count(settings.database_path, "metadata") == 0


def test_missing_stored_pdf_raises_source_error(tmp_path: Path, single_page_pdf: Path) -> None:
    """Category 17b: a metadata row without its PDF fails explicitly."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    (settings.documents_dir / f"{document_id}.pdf").unlink()

    with pytest.raises(DocumentIndexSourceError):
        DocumentIndexService(service).index_document(document_id)


def test_document_deletion_removes_every_index_row(tmp_path: Path, single_page_pdf: Path) -> None:
    """Category 18: deleting a document cascades to page text, chunks and metadata."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)
    assert _row_count(settings.database_path, "chunks") > 0

    service.delete_document(document_id)

    assert _row_count(settings.database_path, "documents") == 0
    assert _row_count(settings.database_path, "page_texts") == 0
    assert _row_count(settings.database_path, "chunks") == 0
    assert _row_count(settings.database_path, "metadata") == 0


def test_explicit_index_cleanup_keeps_the_document_row(
    tmp_path: Path, single_page_pdf: Path
) -> None:
    """Category 18b: ``delete_document_index`` clears only derived rows."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)

    assert service.repository.delete_document_index(document_id) is True
    assert service.repository.get(document_id) is not None
    assert _row_count(settings.database_path, "page_texts") == 0
    assert _row_count(settings.database_path, "chunks") == 0
    assert _row_count(settings.database_path, "metadata") == 0
    assert service.repository.delete_document_index(document_id) is False


def test_foreign_keys_and_check_constraints_are_enforced(
    tmp_path: Path, single_page_pdf: Path
) -> None:
    """Category 19: orphan rows and invalid page numbers are rejected."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    indexer = DocumentIndexService(service)
    indexer.index_document(document_id, max_chars=10)
    existing = indexer.get_document_chunks(document_id)[0]

    with closing(sqlite3.connect(settings.database_path)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO document_page_texts
                    (document_id, page_index, page_number, text, extraction_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(uuid4()), 0, 1, "orphan", TEXT_EXTRACTION_VERSION),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO document_page_texts
                    (document_id, page_index, page_number, text, extraction_version)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(document_id), 1, 5, "bad page number", TEXT_EXTRACTION_VERSION),
            )

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO document_chunks
                    (chunk_id, document_id, page_index, page_number, chunk_index,
                     start_offset, end_offset, text, extraction_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "chk_duplicate_constraint_test",
                    str(document_id),
                    existing.page_index,
                    existing.page_number,
                    existing.chunk_index,
                    existing.start_offset,
                    existing.end_offset,
                    existing.text,
                    existing.extraction_version,
                ),
            )


def test_existing_document_api_contract_does_not_regress(api, pdf_bytes: bytes) -> None:
    """Category 20: upload, list, detail, file, search and delete still work."""
    created = api.client.post(
        DOCUMENTS_URL,
        files={"file": ("regression.pdf", pdf_bytes, "application/pdf")},
    )
    assert created.status_code == 201
    document_id = created.json()["document_id"]

    listing = api.client.get(DOCUMENTS_URL)
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    detail = api.client.get(f"{DOCUMENTS_URL}/{document_id}")
    assert detail.status_code == 200

    file_response = api.client.get(f"{DOCUMENTS_URL}/{document_id}/file")
    assert file_response.status_code == 200
    assert file_response.headers["content-type"] == "application/pdf"

    search = api.client.post(
        f"{DOCUMENTS_URL}/{document_id}/anchors/search",
        json={"query": "Page 1"},
    )
    assert search.status_code == 200
    assert search.json()["total"] == 1

    deleted = api.client.delete(f"{DOCUMENTS_URL}/{document_id}")
    assert deleted.status_code == 204
    assert api.client.get(f"{DOCUMENTS_URL}/{document_id}").status_code == 404


def test_api_delete_cascades_index_rows(api, pdf_bytes: bytes) -> None:
    """Category 20b: an index built through the service is cleaned by API delete."""
    created = api.client.post(
        DOCUMENTS_URL,
        files={"file": ("cascade.pdf", pdf_bytes, "application/pdf")},
    )
    assert created.status_code == 201
    document_id = UUID(created.json()["document_id"])

    indexer = DocumentIndexService(api.app.state.document_service)
    indexer.index_document(document_id, max_chars=10)
    assert _row_count(api.settings.database_path, "chunks") > 0

    assert api.client.delete(f"{DOCUMENTS_URL}/{document_id}").status_code == 204
    assert _row_count(api.settings.database_path, "page_texts") == 0
    assert _row_count(api.settings.database_path, "chunks") == 0
    assert _row_count(api.settings.database_path, "metadata") == 0


def test_m1_01a_chunking_semantics_are_unchanged_by_indexing(tmp_path: Path, cjk_pdf: Path) -> None:
    """Category 21: M1-01A offsets and page text semantics survive persistence."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, cjk_pdf, page_count=1)
    indexer = DocumentIndexService(service)

    indexer.index_document(document_id, max_chars=4)

    pages = indexer.get_document_page_texts(document_id)
    assert pages[0].text == "中文quick证据"
    chunks = indexer.get_document_chunks(document_id)
    assert [chunk.text for chunk in chunks] == ["中文qu", "ick证", "据"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 9),
    ]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_exact_pdf_search_still_works_after_indexing(tmp_path: Path, single_page_pdf: Path) -> None:
    """Category 21b: indexing must not alter PDF anchor search semantics."""
    settings, service = _prepare_service(tmp_path)
    document_id = _store_document(service, settings, single_page_pdf, page_count=1)
    DocumentIndexService(service).index_document(document_id, max_chars=10)

    hits = search_anchors(settings.documents_dir / f"{document_id}.pdf", "fox")

    assert len(hits) == 1
    assert hits[0].page_index == 0
    assert hits[0].matched_text == "fox"


def test_benchmark_dataset_material_remains_intact() -> None:
    """Category 22: the frozen 22-sample benchmark dataset is still present."""
    backend_dir = Path(__file__).resolve().parents[2]
    dataset_dir = backend_dir / "benchmarks" / "datasets" / "v0"
    manifest_path = dataset_dir / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "v0"
    assert len(manifest["samples"]) == 22
    for sample_name in manifest["samples"]:
        assert (dataset_dir / sample_name).is_file()
