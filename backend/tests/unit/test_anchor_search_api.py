"""M0-04B anchor search API tests.

The PDF fixtures live in this package's ``conftest.py`` (generated at runtime
with reportlab/pypdf), while the isolated FastAPI application comes from the
parent ``tests/conftest.py``. No binary PDF is stored in the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_anchor.services import search_anchors

DOCUMENTS_URL = "/api/v1/documents"


def _upload_bytes(client: TestClient, payload: bytes, filename: str = "document.pdf"):
    return client.post(
        DOCUMENTS_URL,
        files={"file": (filename, payload, "application/pdf")},
    )


def _upload_path(client: TestClient, path: Path, filename: str = "document.pdf"):
    return _upload_bytes(client, path.read_bytes(), filename)


def _uploaded_id(client: TestClient, path: Path, filename: str = "document.pdf") -> str:
    response = _upload_path(client, path, filename)
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


def _search(
    client: TestClient,
    document_id: str,
    query: str | None = None,
    *,
    match_case: bool | None = None,
    payload: dict[str, object] | None = None,
):
    body = payload
    if body is None:
        body = {"query": query}
        if match_case is not None:
            body["match_case"] = match_case
    return client.post(f"{DOCUMENTS_URL}/{document_id}/anchors/search", json=body)


def test_search_after_upload_returns_hits(api, single_page_pdf: Path) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    response = _search(api.client, document_id, "fox")

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["query"] == "fox"
    assert body["match_case"] is True
    assert body["total"] == 1
    assert len(body["results"]) == 1


def test_single_hit_returns_text_page_offsets_and_rects(
    api,
    single_page_pdf: Path,
) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    hit = _search(api.client, document_id, "fox").json()["results"][0]

    assert hit["matched_text"] == "fox"
    assert hit["page_index"] == 0
    assert hit["page_number"] == 1
    assert hit["char_start"] == 16
    assert hit["char_count"] == 3
    assert len(hit["rects"]) == 1

    rect = hit["rects"][0]
    assert rect["left"] == pytest.approx(191.536, abs=0.05)
    assert rect["bottom"] == pytest.approx(199.844, abs=0.05)
    assert rect["right"] == pytest.approx(207.256, abs=0.05)
    assert rect["top"] == pytest.approx(208.736, abs=0.05)
    assert hit["bbox"] == rect

    page = hit["page"]
    assert page["page_index"] == 0
    assert page["width_pt"] == 612.0
    assert page["height_pt"] == 792.0
    assert page["rotation"] == 0
    assert page["cropbox"] == [0.0, 0.0, 612.0, 792.0]


def test_repeated_hits_on_same_page_are_all_returned(
    api,
    single_page_pdf: Path,
) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    body = _search(api.client, document_id, "quick").json()

    assert body["total"] == 3
    assert [hit["char_start"] for hit in body["results"]] == [4, 46, 52]
    assert [hit["matched_text"] for hit in body["results"]] == [
        "quick",
        "quick",
        "quick",
    ]
    assert len({tuple(hit["rects"][0].values()) for hit in body["results"]}) == 3


def test_multi_page_hits_use_zero_based_page_index(
    api,
    multi_page_pdf: Path,
) -> None:
    document_id = _uploaded_id(api.client, multi_page_pdf, "multi.pdf")

    body = _search(api.client, document_id, "gamma").json()

    assert body["total"] == 3
    assert [hit["page_index"] for hit in body["results"]] == [0, 1, 2]
    assert [hit["page_number"] for hit in body["results"]] == [1, 2, 3]
    assert [hit["char_start"] for hit in body["results"]] == [18, 18, 18]
    assert [hit["matched_text"] for hit in body["results"]] == [
        "gamma",
        "gamma",
        "gamma",
    ]


def test_match_case_controls_case_sensitivity(api, single_page_pdf: Path) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    sensitive = _search(
        api.client,
        document_id,
        "quick",
        match_case=True,
    ).json()
    insensitive = _search(
        api.client,
        document_id,
        "quick",
        match_case=False,
    ).json()

    assert sensitive["match_case"] is True
    assert sensitive["total"] == 3
    assert [hit["matched_text"] for hit in sensitive["results"]] == [
        "quick",
        "quick",
        "quick",
    ]

    assert insensitive["match_case"] is False
    assert insensitive["total"] == 4
    assert [hit["char_start"] for hit in insensitive["results"]] == [4, 46, 52, 58]
    assert insensitive["results"][-1]["matched_text"] == "QUICK"


@pytest.mark.parametrize("query", ["zzzznotfound", "foxes"])
def test_zero_hits_return_empty_results(
    api,
    single_page_pdf: Path,
    query: str,
) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    response = _search(api.client, document_id, query)

    assert response.status_code == 200
    assert response.json()["total"] == 0
    assert response.json()["results"] == []


@pytest.mark.parametrize("query", ["", " ", "   ", "\n\t"])
def test_empty_or_whitespace_query_returns_422(
    api,
    single_page_pdf: Path,
    query: str,
) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    response = _search(api.client, document_id, query)

    assert response.status_code == 422
    assert "results" not in response.json()


@pytest.mark.parametrize("payload", [{}, {"query": 123}, {"query": None}])
def test_missing_or_non_string_query_returns_422(
    api,
    single_page_pdf: Path,
    payload: dict[str, object],
) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    response = _search(api.client, document_id, payload=payload)

    assert response.status_code == 422
    assert "results" not in response.json()


def test_unknown_document_returns_404(api) -> None:
    response = _search(
        api.client,
        "00000000-0000-4000-8000-000000000000",
        "anything",
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


def test_malformed_document_id_returns_422(api) -> None:
    response = _search(api.client, "not-a-uuid", "anything")

    assert response.status_code == 422


def test_missing_stored_pdf_returns_server_error(api, single_page_pdf: Path) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)
    stored = api.settings.documents_dir / f"{document_id}.pdf"
    stored.unlink()

    response = _search(api.client, document_id, "fox")

    assert response.status_code == 500
    assert response.json() == {"detail": "Stored PDF is missing or unreadable"}
    assert str(api.settings.data_dir) not in response.text
    assert str(api.settings.documents_dir) not in response.text


def test_corrupted_stored_pdf_returns_server_error(api, pdf_bytes: bytes) -> None:
    created = _upload_bytes(api.client, pdf_bytes)
    assert created.status_code == 201
    document_id = created.json()["document_id"]
    (api.settings.documents_dir / f"{document_id}.pdf").write_bytes(b"not a PDF")

    response = _search(api.client, document_id, "Page")

    assert response.status_code == 500
    assert response.json() == {"detail": "Stored PDF could not be searched"}
    assert str(api.settings.data_dir) not in response.text


def test_cross_line_hit_returns_multiple_rects(api, multi_page_pdf: Path) -> None:
    document_id = _uploaded_id(api.client, multi_page_pdf, "multi.pdf")

    body = _search(api.client, document_id, "gamma second").json()

    assert body["total"] == 3
    assert [hit["page_index"] for hit in body["results"]] == [0, 1, 2]
    for hit in body["results"]:
        assert hit["matched_text"] == "gamma\r\nsecond"
        assert hit["char_start"] == 18
        assert hit["char_count"] == 13
        assert len(hit["rects"]) == 2

    upper, lower = body["results"][0]["rects"]
    assert upper["left"] == pytest.approx(201.220, abs=0.05)
    assert upper["bottom"] == pytest.approx(197.372, abs=0.05)
    assert upper["right"] == pytest.approx(240.436, abs=0.05)
    assert upper["top"] == pytest.approx(206.456, abs=0.05)
    assert lower["left"] == pytest.approx(100.384, abs=0.05)
    assert lower["bottom"] == pytest.approx(149.832, abs=0.05)
    assert lower["right"] == pytest.approx(138.004, abs=0.05)
    assert lower["top"] == pytest.approx(158.616, abs=0.05)
    # y grows upwards: the first line is above the second and the gap is not
    # covered by any returned rectangle.
    assert upper["bottom"] > lower["top"]


def test_rect_coordinates_preserve_float_precision(
    api,
    single_page_pdf: Path,
) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf)

    body = _search(api.client, document_id, "brown fox").json()
    hit = body["results"][0]
    expected = search_anchors(single_page_pdf, "brown fox")[0]

    assert hit["rects"] == [
        {
            "left": rect.left,
            "bottom": rect.bottom,
            "right": rect.right,
            "top": rect.top,
        }
        for rect in expected.rects
    ]
    assert hit["bbox"] == {
        "left": expected.bbox.left,
        "bottom": expected.bbox.bottom,
        "right": expected.bbox.right,
        "top": expected.bbox.top,
    }


def test_cropbox_pdf_returns_absolute_user_space_coordinates(
    api,
    cropbox_pdf: Path,
) -> None:
    document_id = _uploaded_id(api.client, cropbox_pdf, "cropbox.pdf")

    hit = _search(api.client, document_id, "quick").json()["results"][0]

    assert hit["page"]["cropbox"] == [50.0, 100.0, 400.0, 500.0]
    assert hit["page"]["width_pt"] == 350.0
    assert hit["page"]["height_pt"] == 400.0

    rect = hit["rects"][0]
    # M0-02 measured these absolute values for text drawn at baseline (100, 200).
    # The API must not subtract the CropBox origin (74.432 / 97.528).
    assert rect["left"] == pytest.approx(124.432, abs=0.05)
    assert rect["bottom"] == pytest.approx(197.528, abs=0.05)
    assert rect["right"] == pytest.approx(152.032, abs=0.05)
    assert rect["top"] == pytest.approx(208.616, abs=0.05)
    assert rect["left"] > hit["page"]["cropbox"][0]
    assert rect["bottom"] > hit["page"]["cropbox"][1]


def test_response_never_contains_server_paths(api, single_page_pdf: Path) -> None:
    document_id = _uploaded_id(api.client, single_page_pdf, "secret.pdf")

    response = _search(api.client, document_id, "fox")

    assert response.status_code == 200
    assert str(api.settings.data_dir) not in response.text
    assert str(api.settings.documents_dir) not in response.text
    assert str(api.settings.temp_dir) not in response.text
    assert str(api.settings.database_path) not in response.text
    assert "secret.pdf" not in response.text
