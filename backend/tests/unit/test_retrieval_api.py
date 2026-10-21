"""M1-02 API tests: POST /api/v1/retrieval/search.

PDFs are generated at runtime. Each test uploads through the public API, then
builds the index through the same service the application exposes on
``app.state``; retrieval itself only ever reads the persisted chunk rows.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from rag_anchor.services import DocumentIndexService

DOCUMENTS_URL = "/api/v1/documents"
RETRIEVAL_URL = "/api/v1/retrieval/search"

HIT_FIELDS = {
    "document_id",
    "chunk_id",
    "page_index",
    "page_number",
    "chunk_index",
    "start_offset",
    "end_offset",
    "match_start",
    "match_end",
    "hit_text",
    "chunk_text",
    "extraction_version",
}

RESPONSE_FIELDS = {
    "query",
    "match_case",
    "document_id",
    "total",
    "limit",
    "offset",
    "results",
}


def _upload(client: TestClient, path: Path, filename: str = "document.pdf") -> str:
    response = client.post(
        DOCUMENTS_URL,
        files={"file": (filename, path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


def _index(api, document_id: str) -> None:
    """Index with the service defaults so each test page fits in one chunk."""
    service = api.app.state.document_service
    DocumentIndexService(service).index_document(UUID(document_id))


def _search(client: TestClient, payload: dict[str, object]):
    return client.post(RETRIEVAL_URL, json=payload)


def test_search_returns_full_provenance_without_pdf_geometry(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    _index(api, document_id)

    response = _search(api.client, {"query": "fox"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["query"] == "fox"
    assert body["match_case"] is True
    assert body["document_id"] is None
    assert body["total"] == 1
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert len(body["results"]) == 1

    hit = body["results"][0]
    assert set(hit) == HIT_FIELDS
    assert hit["document_id"] == document_id
    assert hit["chunk_id"].startswith("chk_")
    assert hit["page_index"] == 0
    assert hit["page_number"] == 1
    assert hit["chunk_index"] >= 0
    assert hit["start_offset"] <= hit["match_start"]
    assert hit["match_end"] <= hit["end_offset"]
    assert hit["hit_text"] == "fox"
    assert (
        hit["chunk_text"][
            hit["match_start"] - hit["start_offset"] : hit["match_end"] - hit["start_offset"]
        ]
        == "fox"
    )
    assert hit["extraction_version"]

    # The chunk retrieval contract never exposes PDF user-space geometry.
    assert not {"rects", "bbox", "page"} & set(hit)


def test_search_without_hits_returns_empty_page(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    _index(api, document_id)

    response = _search(api.client, {"query": "definitely-not-present"})

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["results"] == []


def test_search_unknown_document_returns_404(api) -> None:
    response = _search(
        api.client,
        {"query": "alpha", "document_id": "00000000-0000-0000-0000-000000000000"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "ok", "limit": 0},
        {"query": "ok", "limit": 101},
        {"query": "ok", "offset": -1},
        {},
    ],
)
def test_invalid_payload_is_rejected(api, payload: dict[str, object]) -> None:
    response = _search(api.client, payload)

    assert response.status_code == 422, response.text


def test_multi_document_search_and_filter(api, single_page_pdf: Path) -> None:
    first = _upload(api.client, single_page_pdf, "first.pdf")
    second = _upload(api.client, single_page_pdf, "second.pdf")
    _index(api, first)
    _index(api, second)

    everything = _search(api.client, {"query": "fox"})
    assert everything.status_code == 200
    assert everything.json()["total"] == 2
    assert {hit["document_id"] for hit in everything.json()["results"]} == {first, second}

    filtered = _search(api.client, {"query": "fox", "document_id": second})
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["results"][0]["document_id"] == second


def test_case_insensitive_search_keeps_original_text(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    _index(api, document_id)

    lower = _search(api.client, {"query": "quick", "match_case": False}).json()
    upper = _search(api.client, {"query": "QUICK", "match_case": False}).json()
    sensitive = _search(api.client, {"query": "QUICK"}).json()

    assert lower["match_case"] is False
    assert lower["total"] == 4
    assert upper["total"] == 4
    assert [hit["hit_text"] for hit in upper["results"]] == [
        "quick",
        "quick",
        "quick",
        "QUICK",
    ]
    assert sensitive["total"] == 1
    assert sensitive["results"][0]["hit_text"] == "QUICK"


def test_pagination_reports_unpaginated_total(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    _index(api, document_id)

    first = _search(api.client, {"query": "quick", "limit": 2, "offset": 0}).json()
    second = _search(api.client, {"query": "quick", "limit": 2, "offset": 2}).json()

    assert first["total"] == 3
    assert len(first["results"]) == 2
    assert second["total"] == 3
    assert len(second["results"]) == 1
    assert first["results"] != second["results"]


def test_deleted_document_is_excluded(api, single_page_pdf: Path) -> None:
    kept = _upload(api.client, single_page_pdf, "kept.pdf")
    removed = _upload(api.client, single_page_pdf, "removed.pdf")
    _index(api, kept)
    _index(api, removed)
    assert _search(api.client, {"query": "fox"}).json()["total"] == 2

    delete = api.client.delete(f"{DOCUMENTS_URL}/{removed}")
    assert delete.status_code == 204, delete.text

    after_delete = _search(api.client, {"query": "fox"}).json()
    assert after_delete["total"] == 1
    assert after_delete["results"][0]["document_id"] == kept

    filtered = _search(api.client, {"query": "fox", "document_id": removed})
    assert filtered.status_code == 404


def test_restart_reads_persisted_chunks(api_factory, single_page_pdf: Path) -> None:
    first = api_factory()
    document_id = _upload(first.client, single_page_pdf)
    _index(first, document_id)

    restarted = api_factory()

    response = _search(restarted.client, {"query": "fox"})
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["results"][0]["document_id"] == document_id


def test_search_does_not_change_index_rows(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)
    _index(api, document_id)
    service = api.app.state.document_service
    before = service.repository.get_document_chunks(UUID(document_id))

    _search(api.client, {"query": "quick", "match_case": False})
    _search(api.client, {"query": "fox"})

    after = service.repository.get_document_chunks(UUID(document_id))
    assert after == before
