"""M1-03A API tests: POST /api/v1/documents/{document_id}/index.

Every test uploads a runtime-generated PDF through the public HTTP API and
triggers indexing through the same public API. The service layer is only read
directly when a test needs to inspect persisted rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient

DOCUMENTS_URL = "/api/v1/documents"
RETRIEVAL_URL = "/api/v1/retrieval/search"

INDEX_FIELDS = {
    "document_id",
    "fingerprint",
    "page_count",
    "chunk_count",
    "filtered_chunk_count",
    "indexed_at",
}


def _upload(client: TestClient, path: Path, filename: str = "document.pdf") -> str:
    response = client.post(
        DOCUMENTS_URL,
        files={"file": (filename, path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


def _index(client: TestClient, document_id: str):
    return client.post(f"{DOCUMENTS_URL}/{document_id}/index")


def _search(client: TestClient, payload: dict[str, object]):
    return client.post(RETRIEVAL_URL, json=payload)


def _chunk_ids(api, document_id: str) -> list[str]:
    chunks = api.app.state.document_service.repository.get_document_chunks(UUID(document_id))
    return [chunk.chunk_id for chunk in chunks]


def test_index_returns_statistics_and_enables_retrieval(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)

    response = _index(api.client, document_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == INDEX_FIELDS
    assert body["document_id"] == document_id
    assert body["page_count"] == 1
    assert body["chunk_count"] >= 1
    assert body["filtered_chunk_count"] >= 0
    assert len(body["fingerprint"]) == 64
    assert datetime.fromisoformat(body["indexed_at"].replace("Z", "+00:00"))

    search = _search(api.client, {"query": "fox", "document_id": document_id})
    assert search.status_code == 200, search.text
    results = search.json()["results"]
    assert len(results) == 1
    hit = results[0]
    assert hit["document_id"] == document_id
    assert hit["page_index"] == 0
    assert hit["hit_text"] == "fox"

    pages = api.app.state.document_service.repository.get_document_page_texts(UUID(document_id))
    page = next(item for item in pages if item.page_index == hit["page_index"])
    assert page.text[hit["match_start"] : hit["match_end"]] == hit["hit_text"]


def test_repeated_index_is_idempotent(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)

    first = _index(api.client, document_id)
    assert first.status_code == 200, first.text
    first_ids = _chunk_ids(api, document_id)
    assert first_ids

    second = _index(api.client, document_id)
    assert second.status_code == 200, second.text

    assert _chunk_ids(api, document_id) == first_ids
    assert second.json()["page_count"] == first.json()["page_count"]
    assert second.json()["chunk_count"] == first.json()["chunk_count"]
    assert second.json()["filtered_chunk_count"] == first.json()["filtered_chunk_count"]
    assert second.json()["fingerprint"] == first.json()["fingerprint"]

    search = _search(api.client, {"query": "fox", "document_id": document_id})
    assert search.json()["total"] == 1


def test_index_unknown_document_returns_404(api) -> None:
    response = _index(api.client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_index_invalid_uuid_returns_422(api) -> None:
    response = _index(api.client, "not-a-uuid")

    assert response.status_code == 422


def test_index_missing_stored_pdf_returns_500_without_path(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    stored_pdf = api.settings.documents_dir / f"{document_id}.pdf"
    assert stored_pdf.is_file()
    stored_pdf.unlink()

    response = _index(api.client, document_id)

    assert response.status_code == 500
    assert response.json()["detail"] == "Stored PDF is missing or unreadable"
    assert str(api.settings.data_dir) not in response.text


def test_failed_rebuild_keeps_previous_index(api, single_page_pdf: Path, monkeypatch) -> None:
    document_id = _upload(api.client, single_page_pdf)
    first = _index(api.client, document_id)
    assert first.status_code == 200, first.text

    repository = api.app.state.document_service.repository
    pages_before = repository.get_document_page_texts(UUID(document_id))
    chunks_before = repository.get_document_chunks(UUID(document_id))
    metadata_before = repository.get_document_index_metadata(UUID(document_id))
    assert pages_before
    assert chunks_before
    assert metadata_before is not None

    def _fail(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(repository, "replace_document_index", _fail)

    response = _index(api.client, document_id)

    assert response.status_code == 500
    assert response.json()["detail"] == "Document index could not be stored"
    assert repository.get_document_page_texts(UUID(document_id)) == pages_before
    assert repository.get_document_chunks(UUID(document_id)) == chunks_before
    assert repository.get_document_index_metadata(UUID(document_id)) == metadata_before


def test_deleted_document_cannot_be_indexed(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    assert _index(api.client, document_id).status_code == 200

    delete = api.client.delete(f"{DOCUMENTS_URL}/{document_id}")
    assert delete.status_code == 204, delete.text

    response = _index(api.client, document_id)
    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"

    search = _search(api.client, {"query": "fox", "document_id": document_id})
    assert search.status_code == 404


def test_index_survives_restart(api_factory, single_page_pdf: Path) -> None:
    first = api_factory()
    document_id = _upload(first.client, single_page_pdf)
    indexed = _index(first.client, document_id)
    assert indexed.status_code == 200, indexed.text

    restarted = api_factory()

    search = _search(restarted.client, {"query": "fox"})
    assert search.status_code == 200
    assert search.json()["total"] == 1
    assert search.json()["results"][0]["document_id"] == document_id

    reindexed = _index(restarted.client, document_id)
    assert reindexed.status_code == 200, reindexed.text
    assert reindexed.json()["fingerprint"] == indexed.json()["fingerprint"]
