"""Integration tests for the read-only document file endpoint (M0-05A).

PDFs are generated at runtime by the shared ``pdf_builder`` fixture; no binary
document is committed to the repository and every test runs against a temporary
data directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from tests.conftest import ApiContext

_FIXED_DISPOSITION = 'inline; filename="document.pdf"'


def _upload(client: TestClient, data: bytes, filename: str = "evidence.pdf") -> dict[str, object]:
    """Upload one PDF and return its metadata."""
    response = client.post(
        "/api/v1/documents",
        files={"file": (filename, data, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _file_url(document_id: object) -> str:
    return f"/api/v1/documents/{document_id}/file"


def test_get_document_file_returns_uploaded_pdf_bytes(api: ApiContext, pdf_bytes: bytes) -> None:
    """A stored document can be read back byte-for-byte as a PDF."""
    metadata = _upload(api.client, pdf_bytes)
    response = api.client.get(_file_url(metadata["document_id"]))

    assert response.status_code == 200
    assert response.content == pdf_bytes
    assert response.content.startswith(b"%PDF-")
    assert response.content.rstrip().endswith(b"%%EOF")
    assert int(response.headers["content-length"]) == len(pdf_bytes)


def test_file_response_headers_request_inline_pdf(api: ApiContext, pdf_bytes: bytes) -> None:
    """The response advertises an inline PDF with a safe fixed filename."""
    metadata = _upload(api.client, pdf_bytes, filename="../../sensitive name.pdf")
    response = api.client.get(_file_url(metadata["document_id"]))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["content-disposition"] == _FIXED_DISPOSITION
    # The client-supplied name must never reach the response headers.
    assert "sensitive" not in response.headers["content-disposition"]
    assert ".." not in response.headers["content-disposition"]


def test_missing_stored_pdf_returns_500_without_leaking_paths(
    api: ApiContext, pdf_bytes: bytes
) -> None:
    """Known metadata with a missing file is a controlled server error."""
    metadata = _upload(api.client, pdf_bytes)
    stored_path = api.settings.documents_dir / f"{metadata['document_id']}.pdf"
    assert stored_path.is_file()
    stored_path.unlink()

    response = api.client.get(_file_url(metadata["document_id"]))

    assert response.status_code == 500
    assert response.json() == {"detail": "Stored PDF is missing or unreadable"}
    body = response.text
    assert str(api.settings.data_dir) not in body
    assert str(api.settings.documents_dir) not in body
    assert str(api.settings.database_path) not in body
    assert str(api.settings.temp_dir) not in body
    assert "metadata.sqlite3" not in body


def test_unknown_document_returns_404(api: ApiContext) -> None:
    """A syntactically valid but unknown UUID returns 404."""
    response = api.client.get("/api/v1/documents/00000000-0000-0000-0000-000000000000/file")

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


@pytest.mark.parametrize(
    "document_id",
    ["not-a-uuid", "12345", "00000000-0000-0000-0000-00000000000z"],
)
def test_malformed_document_id_returns_422(api: ApiContext, document_id: str) -> None:
    """Invalid UUIDs keep the existing FastAPI 422 behaviour."""
    response = api.client.get(_file_url(document_id))

    assert response.status_code == 422


def test_file_read_does_not_modify_document_state(api: ApiContext, pdf_bytes: bytes) -> None:
    """Reading a file leaves metadata, the document list and bytes untouched."""
    metadata = _upload(api.client, pdf_bytes)
    document_id = metadata["document_id"]
    stored_path = api.settings.documents_dir / f"{document_id}.pdf"
    bytes_before = stored_path.read_bytes()

    list_before = api.client.get("/api/v1/documents").json()
    detail_before = api.client.get(f"/api/v1/documents/{document_id}").json()

    first = api.client.get(_file_url(document_id))
    second = api.client.get(_file_url(document_id))

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.content == second.content == pdf_bytes

    list_after = api.client.get("/api/v1/documents").json()
    detail_after = api.client.get(f"/api/v1/documents/{document_id}").json()

    assert list_after == list_before
    assert detail_after == detail_before
    assert stored_path.is_file()
    assert stored_path.read_bytes() == bytes_before


def test_file_endpoint_ignores_client_filename_for_storage(
    api: ApiContext, pdf_builder: Callable[..., bytes]
) -> None:
    """A traversal-shaped upload name cannot redirect which bytes are served."""
    data = pdf_builder(page_count=2)
    metadata = _upload(api.client, data, filename="../../../../etc/passwd")

    stored_files = sorted(api.settings.documents_dir.iterdir())
    assert [path.name for path in stored_files] == [f"{metadata['document_id']}.pdf"]

    response = api.client.get(_file_url(metadata["document_id"]))
    assert response.status_code == 200
    assert response.content == data
    assert response.headers["content-disposition"] == _FIXED_DISPOSITION


def test_file_endpoint_is_present_in_openapi(api: ApiContext) -> None:
    """The file endpoint is documented with the PDF media type."""
    schema = api.client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/v1/documents/{document_id}/file" in paths

    operation = paths["/api/v1/documents/{document_id}/file"]["get"]
    assert "application/pdf" in operation["responses"]["200"]["content"]


def test_response_headers_never_include_server_paths(api: ApiContext, pdf_bytes: bytes) -> None:
    """Neither the PDF body nor the headers may expose the data directory."""
    metadata = _upload(api.client, pdf_bytes)
    response = api.client.get(_file_url(metadata["document_id"]))

    serialized_headers = json.dumps(dict(response.headers))
    forbidden = (
        str(api.settings.data_dir),
        str(api.settings.documents_dir),
        str(api.settings.temp_dir),
        str(api.settings.database_path),
        "metadata.sqlite3",
    )
    for value in forbidden:
        assert value not in serialized_headers
        assert value.encode() not in response.content

    assert Path(str(api.settings.documents_dir)).is_dir()
