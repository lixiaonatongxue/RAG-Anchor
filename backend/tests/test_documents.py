"""M0-04A document API tests."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_anchor.config import DEFAULT_MAX_UPLOAD_BYTES, Settings, default_data_dir
from rag_anchor.main import create_app

DOCUMENTS_URL = "/api/v1/documents"


def _upload(client: TestClient, filename: str, payload: bytes):
    return client.post(
        DOCUMENTS_URL,
        files={"file": (filename, payload, "application/pdf")},
    )


def _stored_files(context) -> list[Path]:
    return sorted(context.settings.documents_dir.iterdir())


def test_upload_valid_pdf_returns_metadata(api, pdf_builder) -> None:
    payload = pdf_builder(page_count=3)

    response = _upload(api.client, "report.pdf", payload)

    assert response.status_code == 201
    body = response.json()
    document_id = uuid.UUID(body["document_id"])
    assert body["filename"] == "report.pdf"
    assert body["size_bytes"] == len(payload)
    assert body["page_count"] == 3
    assert "path" not in body
    assert "stored_path" not in body
    datetime.fromisoformat(body["created_at"])

    stored = api.settings.documents_dir / f"{document_id}.pdf"
    assert stored.is_file()
    assert stored.read_bytes() == payload


def test_duplicate_uploads_get_distinct_ids(api, pdf_bytes: bytes) -> None:
    first = _upload(api.client, "same.pdf", pdf_bytes)
    second = _upload(api.client, "same.pdf", pdf_bytes)

    assert first.status_code == 201
    assert second.status_code == 201
    first_id = first.json()["document_id"]
    second_id = second.json()["document_id"]
    assert first_id != second_id
    assert (api.settings.documents_dir / f"{first_id}.pdf").is_file()
    assert (api.settings.documents_dir / f"{second_id}.pdf").is_file()

    listing = api.client.get(DOCUMENTS_URL).json()
    assert listing["total"] == 2


@pytest.mark.parametrize(
    "payload",
    [
        b"this is definitely not a PDF",
        b"%PDF-1.7\nthis body is not a real PDF document",
    ],
)
def test_invalid_pdf_is_rejected_without_side_effects(
    api,
    payload: bytes,
) -> None:
    response = _upload(api.client, "broken.pdf", payload)

    assert response.status_code == 400
    assert response.json()["detail"]
    assert _stored_files(api) == []
    assert list(api.settings.temp_dir.iterdir()) == []
    assert api.client.get(DOCUMENTS_URL).json()["total"] == 0


def test_oversized_upload_returns_413_and_cleans_up(api_factory, pdf_builder) -> None:
    context = api_factory(max_upload_bytes=100)
    payload = pdf_builder()

    response = _upload(context.client, "too-big.pdf", payload)

    assert response.status_code == 413
    assert "maximum size" in response.json()["detail"]
    assert _stored_files(context) == []
    assert list(context.settings.temp_dir.iterdir()) == []
    assert context.client.get(DOCUMENTS_URL).json()["total"] == 0


@pytest.mark.parametrize(
    ("upload_name", "expected_name"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\windows\\evil.pdf", "evil.pdf"),
        ("/absolute/path/paper.pdf", "paper.pdf"),
    ],
)
def test_filename_cannot_escape_storage_directory(
    api,
    pdf_bytes: bytes,
    upload_name: str,
    expected_name: str,
) -> None:
    response = _upload(api.client, upload_name, pdf_bytes)

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == expected_name
    stored = _stored_files(api)
    assert len(stored) == 1
    assert stored[0].name == f"{body['document_id']}.pdf"
    assert stored[0].parent == api.settings.documents_dir


def test_list_is_paginated_and_stable(api, pdf_builder) -> None:
    uploaded_ids = []
    for index in range(3):
        response = _upload(api.client, f"doc-{index}.pdf", pdf_builder(page_count=index + 1))
        assert response.status_code == 201
        uploaded_ids.append(response.json()["document_id"])

    listing = api.client.get(DOCUMENTS_URL)
    assert listing.status_code == 200
    body = listing.json()
    assert body["total"] == 3
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert {item["document_id"] for item in body["items"]} == set(uploaded_ids)

    # Mirror the SQL ordering: created_at DESC, then document_id ASC.
    expected = sorted(body["items"], key=lambda item: item["document_id"])
    expected = sorted(expected, key=lambda item: item["created_at"], reverse=True)
    assert [item["document_id"] for item in body["items"]] == [
        item["document_id"] for item in expected
    ]

    page = api.client.get(DOCUMENTS_URL, params={"limit": 2, "offset": 1})
    assert page.status_code == 200
    page_body = page.json()
    assert page_body["total"] == 3
    assert page_body["limit"] == 2
    assert page_body["offset"] == 1
    assert [item["document_id"] for item in page_body["items"]] == [
        item["document_id"] for item in expected[1:3]
    ]


def test_list_rejects_excessive_limit(api) -> None:
    response = api.client.get(DOCUMENTS_URL, params={"limit": 101})

    assert response.status_code == 422


def test_get_document_and_unknown_id(api, pdf_bytes: bytes) -> None:
    created = _upload(api.client, "lookup.pdf", pdf_bytes).json()

    found = api.client.get(f"{DOCUMENTS_URL}/{created['document_id']}")
    assert found.status_code == 200
    assert found.json() == created

    missing = api.client.get(f"{DOCUMENTS_URL}/{uuid.uuid4()}")
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Document not found"}

    malformed = api.client.get(f"{DOCUMENTS_URL}/not-a-uuid")
    assert malformed.status_code == 422


def test_delete_removes_file_and_metadata(api, pdf_bytes: bytes) -> None:
    created = _upload(api.client, "delete-me.pdf", pdf_bytes).json()
    document_id = created["document_id"]
    stored = api.settings.documents_dir / f"{document_id}.pdf"
    assert stored.is_file()

    deleted = api.client.delete(f"{DOCUMENTS_URL}/{document_id}")
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert not stored.exists()

    assert api.client.get(f"{DOCUMENTS_URL}/{document_id}").status_code == 404
    assert api.client.delete(f"{DOCUMENTS_URL}/{document_id}").status_code == 404
    assert api.client.get(DOCUMENTS_URL).json()["total"] == 0


def test_metadata_survives_application_restart(tmp_path: Path, pdf_bytes: bytes) -> None:
    data_dir = tmp_path / "shared-data"
    settings = Settings(data_dir=data_dir)

    first_app = create_app(settings)
    with TestClient(first_app) as first_client:
        created = _upload(first_client, "persisted.pdf", pdf_bytes).json()

    second_app = create_app(Settings(data_dir=data_dir))
    with TestClient(second_app) as second_client:
        detail = second_client.get(f"{DOCUMENTS_URL}/{created['document_id']}")
        assert detail.status_code == 200
        assert detail.json()["filename"] == "persisted.pdf"

        listing = second_client.get(DOCUMENTS_URL).json()
        assert listing["total"] == 1
        assert listing["items"][0]["document_id"] == created["document_id"]


def test_metadata_failure_removes_stored_file(
    api,
    pdf_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create(record) -> None:
        raise sqlite3.OperationalError("simulated database failure")

    monkeypatch.setattr(api.app.state.document_service.repository, "create", fail_create)

    response = _upload(api.client, "will-fail.pdf", pdf_bytes)

    assert response.status_code == 500
    assert _stored_files(api) == []
    assert list(api.settings.temp_dir.iterdir()) == []
    assert api.client.get(DOCUMENTS_URL).json()["total"] == 0


def test_move_failure_removes_temporary_file(
    api,
    pdf_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        "rag_anchor.services.document_service.os.replace",
        fail_replace,
    )

    response = _upload(api.client, "cannot-move.pdf", pdf_bytes)

    assert response.status_code == 500
    assert _stored_files(api) == []
    assert list(api.settings.temp_dir.iterdir()) == []
    assert api.client.get(DOCUMENTS_URL).json()["total"] == 0


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configured = tmp_path / "configured"

    monkeypatch.delenv("RAG_ANCHOR_DATA_DIR", raising=False)
    monkeypatch.delenv("RAG_ANCHOR_MAX_UPLOAD_BYTES", raising=False)
    default_settings = Settings.from_env()
    assert default_settings.data_dir == default_data_dir()
    assert default_settings.max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES

    monkeypatch.setenv("RAG_ANCHOR_DATA_DIR", str(configured))
    monkeypatch.setenv("RAG_ANCHOR_MAX_UPLOAD_BYTES", "4096")
    custom = Settings.from_env()
    assert custom.data_dir == configured
    assert custom.max_upload_bytes == 4096

    monkeypatch.setenv("RAG_ANCHOR_MAX_UPLOAD_BYTES", "not-a-number")
    with pytest.raises(ValueError, match="must be an integer"):
        Settings.from_env()


def test_default_data_dir_is_independent_of_cwd(tmp_path: Path) -> None:
    expected = Path(__file__).resolve().parents[1] / ".data"
    assert default_data_dir() == expected
