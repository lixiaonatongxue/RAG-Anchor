"""M1-03B API tests: POST /api/v1/documents/{document_id}/evidence/locate.

Every test drives the public HTTP surface only: upload -> index -> retrieval
search -> evidence locate. Rectangle assertions check shape, containment and
derivation from PDFium-returned rects; they never hardcode coordinate values.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

DOCUMENTS_URL = "/api/v1/documents"
RETRIEVAL_URL = "/api/v1/retrieval/search"

_DEJAVU_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
)

LOCATE_FIELDS = {"document_id", "total", "results"}
RESULT_FIELDS = {
    "page_index",
    "page_number",
    "match_start",
    "match_end",
    "hit_text",
    "chunk_id",
    "char_start",
    "char_count",
    "matched_text",
    "rects",
    "bbox",
    "page",
}
RECT_FIELDS = {"left", "bottom", "right", "top"}
PAGE_FIELDS = {"page_index", "width_pt", "height_pt", "rotation", "cropbox"}


def _upload(client: TestClient, path: Path, filename: str = "document.pdf") -> str:
    response = client.post(
        DOCUMENTS_URL,
        files={"file": (filename, path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


def _index(client: TestClient, document_id: str) -> None:
    response = client.post(f"{DOCUMENTS_URL}/{document_id}/index")
    assert response.status_code == 200, response.text


def _search(client: TestClient, payload: dict[str, object]):
    return client.post(RETRIEVAL_URL, json=payload)


def _hits(client: TestClient, query: str, document_id: str) -> list[dict]:
    response = _search(client, {"query": query, "document_id": document_id})
    assert response.status_code == 200, response.text
    return response.json()["results"]


def _items_from_hits(hits: list[dict]) -> list[dict[str, object]]:
    return [
        {
            "page_index": hit["page_index"],
            "match_start": hit["match_start"],
            "match_end": hit["match_end"],
            "hit_text": hit["hit_text"],
            "chunk_id": hit["chunk_id"],
        }
        for hit in hits
    ]


def _locate(client: TestClient, document_id: str, items: list[dict[str, object]]):
    return client.post(
        f"{DOCUMENTS_URL}/{document_id}/evidence/locate",
        json={"items": items},
    )


def _indexed_document(api, path: Path) -> str:
    document_id = _upload(api.client, path)
    _index(api.client, document_id)
    return document_id


def _non_bmp_font() -> str:
    """Register a font with supplementary-plane coverage, or skip the test."""
    for candidate in _DEJAVU_CANDIDATES:
        if not candidate.is_file():
            continue
        name = "DejaVuEvidenceApiProbe"
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(candidate)))
        return name
    pytest.skip("No font with supplementary-plane coverage is available")


def test_end_to_end_retrieval_hits_locate_real_geometry(api, single_page_pdf: Path) -> None:
    """The formal M1-03B E2E: HTTP upload -> index -> search -> locate."""
    document_id = _indexed_document(api, single_page_pdf)
    hits = _hits(api.client, "quick", document_id)
    assert len(hits) == 3

    response = _locate(api.client, document_id, _items_from_hits(hits))

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == LOCATE_FIELDS
    assert body["document_id"] == document_id
    assert body["total"] == 3
    assert len(body["results"]) == 3

    for hit, result in zip(hits, body["results"], strict=True):
        assert set(result) == RESULT_FIELDS
        assert result["page_index"] == hit["page_index"] == 0
        assert result["page_number"] == hit["page_number"] == 1
        assert result["match_start"] == hit["match_start"]
        assert result["match_end"] == hit["match_end"]
        assert result["hit_text"] == hit["hit_text"] == "quick"
        assert result["chunk_id"] == hit["chunk_id"]
        assert result["matched_text"] == "quick"
        # The page is pure BMP text, so PDFium indices align with Python offsets.
        assert result["char_start"] == hit["match_start"]
        assert result["char_count"] == 5

        assert result["rects"]
        for rect in result["rects"]:
            assert set(rect) == RECT_FIELDS
            assert rect["left"] < rect["right"]
            assert rect["bottom"] < rect["top"]

        bbox = result["bbox"]
        assert bbox is not None
        assert bbox == {
            "left": min(rect["left"] for rect in result["rects"]),
            "bottom": min(rect["bottom"] for rect in result["rects"]),
            "right": max(rect["right"] for rect in result["rects"]),
            "top": max(rect["top"] for rect in result["rects"]),
        }

        page = result["page"]
        assert set(page) == PAGE_FIELDS
        assert page["page_index"] == 0
        assert page["width_pt"] == pytest.approx(612.0)
        assert page["height_pt"] == pytest.approx(792.0)
        assert page["rotation"] == 0
        assert page["cropbox"] == pytest.approx((0.0, 0.0, 612.0, 792.0))
        assert 0.0 <= bbox["left"] < bbox["right"] <= page["width_pt"]
        assert 0.0 <= bbox["bottom"] < bbox["top"] <= page["height_pt"]

    # Distinct occurrences keep distinct geometry.
    first_rects = {tuple(sorted(r.items())) for r in body["results"][0]["rects"]}
    second_rects = {tuple(sorted(r.items())) for r in body["results"][1]["rects"]}
    assert first_rects != second_rects


def test_multi_page_batch_locates_the_requested_pages(api, multi_page_pdf: Path) -> None:
    document_id = _indexed_document(api, multi_page_pdf)
    hits = _hits(api.client, "quick", document_id)
    assert [hit["page_index"] for hit in hits] == [0, 1, 2]

    response = _locate(api.client, document_id, _items_from_hits(hits))

    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [item["page_index"] for item in results] == [0, 1, 2]
    assert [item["page_number"] for item in results] == [1, 2, 3]
    assert all(item["hit_text"] == "quick" for item in results)
    assert all(item["rects"] for item in results)


def test_batch_order_and_duplicates_are_preserved(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    hits = _hits(api.client, "quick", document_id)
    items = [hits[1], hits[0], hits[1]]

    response = _locate(api.client, document_id, _items_from_hits(items))

    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert response.json()["total"] == 3
    assert [item["match_start"] for item in results] == [
        items[0]["match_start"],
        items[1]["match_start"],
        items[2]["match_start"],
    ]
    assert results[0]["rects"] == results[2]["rects"]


def test_cross_line_span_returns_one_rectangle_per_line(api, pdf_factory) -> None:
    path = pdf_factory(
        "crlf_api.pdf",
        [[(100.0, 700.0, "alpha quick"), (100.0, 650.0, "brown beta")]],
    )
    document_id = _indexed_document(api, path)
    pages = api.app.state.evidence_service.repository.get_document_page_texts(UUID(document_id))
    needle = "quick\r\nbrown"
    start = pages[0].text.index(needle)

    response = _locate(
        api.client,
        document_id,
        [
            {
                "page_index": 0,
                "match_start": start,
                "match_end": start + len(needle),
                "hit_text": needle,
            }
        ],
    )

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert len(result["rects"]) == 2
    assert result["rects"][0]["bottom"] > result["rects"][1]["bottom"]
    assert result["bbox"] is not None


def test_non_bmp_document_locates_with_offset_gap(api, non_bmp_pdf: Path) -> None:
    document_id = _indexed_document(api, non_bmp_pdf)
    hits = _hits(api.client, "quick", document_id)
    assert len(hits) == 1
    assert hits[0]["match_start"] == 0

    response = _locate(api.client, document_id, _items_from_hits(hits))

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    # The supplementary-plane character is one PDFium index but zero Python
    # characters, so the PDFium start index differs from the stored offset.
    assert result["char_start"] == 1
    assert result["char_start"] != result["match_start"]
    assert result["char_count"] == 5
    assert result["matched_text"] == "quick"
    assert result["rects"]


def test_span_enclosing_non_bmp_returns_409(api, pdf_factory) -> None:
    path = pdf_factory(
        "inner_non_bmp_api.pdf",
        [[(100.0, 700.0, "a\U0001d11eb quick")]],
        font_name=_non_bmp_font(),
    )
    document_id = _indexed_document(api, path)
    pages = api.app.state.evidence_service.repository.get_document_page_texts(UUID(document_id))
    assert pages[0].text == "ab quick"

    response = _locate(
        api.client,
        document_id,
        [{"page_index": 0, "match_start": 0, "match_end": 2, "hit_text": "ab"}],
    )

    assert response.status_code == 409, response.text
    assert "produces no text" in response.json()["detail"]


def test_unknown_document_returns_404(api) -> None:
    response = _locate(
        api.client,
        "00000000-0000-0000-0000-000000000000",
        [{"page_index": 0, "match_start": 0, "match_end": 1, "hit_text": "x"}],
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


@pytest.mark.parametrize(
    "payload",
    [
        {"items": []},
        {"items": [{"page_index": 0, "match_start": 0, "match_end": 1, "hit_text": "T"}] * 21},
        {"items": [{"page_index": 0, "match_start": 2, "match_end": 2, "hit_text": "ab"}]},
        {"items": [{"page_index": 0, "match_start": 0, "match_end": 5, "hit_text": "ab"}]},
        {"items": [{"page_index": -1, "match_start": 0, "match_end": 1, "hit_text": "T"}]},
        {"items": [{}]},
    ],
)
def test_invalid_payload_is_rejected_by_the_schema(api, payload: dict[str, object]) -> None:
    response = api.client.post(
        f"{DOCUMENTS_URL}/00000000-0000-0000-0000-000000000000/evidence/locate",
        json=payload,
    )

    assert response.status_code == 422, response.text


def test_hit_text_mismatch_returns_422(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    response = _locate(
        api.client,
        document_id,
        [{"page_index": 0, "match_start": 0, "match_end": 3, "hit_text": "Xhe"}],
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "hit_text does not match the stored page text"


def test_page_outside_document_returns_422(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    response = _locate(
        api.client,
        document_id,
        [{"page_index": 5, "match_start": 0, "match_end": 1, "hit_text": "T"}],
    )

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "page_index 5 is outside the indexed document"


def test_unknown_chunk_id_returns_422(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    item = _items_from_hits(_hits(api.client, "fox", document_id))[0]
    item["chunk_id"] = "chk_does_not_exist"

    response = _locate(api.client, document_id, [item])

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "chunk_id does not exist"


def test_unindexed_document_returns_409(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)

    response = _locate(
        api.client,
        document_id,
        [{"page_index": 0, "match_start": 0, "match_end": 1, "hit_text": "T"}],
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Document has not been indexed"


def test_tampered_pdf_returns_409(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    items = _items_from_hits(_hits(api.client, "fox", document_id))
    stored_pdf = api.settings.documents_dir / f"{document_id}.pdf"
    stored_pdf.write_bytes(stored_pdf.read_bytes() + b"\n% tampered after indexing\n")

    response = _locate(api.client, document_id, items)

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Stored PDF does not match the index"


def test_missing_pdf_returns_500_without_leaking_paths(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    items = _items_from_hits(_hits(api.client, "fox", document_id))
    (api.settings.documents_dir / f"{document_id}.pdf").unlink()

    response = _locate(api.client, document_id, items)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Stored PDF is missing or unreadable"
    assert str(api.settings.data_dir) not in response.text


def test_locate_does_not_change_index_rows(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    repository = api.app.state.document_service.repository
    document_uuid = UUID(document_id)
    chunks_before = repository.get_document_chunks(document_uuid)
    pages_before = repository.get_document_page_texts(document_uuid)
    metadata_before = repository.get_document_index_metadata(document_uuid)

    _locate(api.client, document_id, _items_from_hits(_hits(api.client, "quick", document_id)))

    assert repository.get_document_chunks(document_uuid) == chunks_before
    assert repository.get_document_page_texts(document_uuid) == pages_before
    assert repository.get_document_index_metadata(document_uuid) == metadata_before


def test_restart_locates_the_persisted_index(api_factory, single_page_pdf: Path) -> None:
    first = api_factory()
    document_id = _upload(first.client, single_page_pdf)
    _index(first.client, document_id)
    hits = _hits(first.client, "quick", document_id)
    items = _items_from_hits(hits)

    restarted = api_factory()
    response = _locate(restarted.client, document_id, items)

    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [item["match_start"] for item in results] == [hit["match_start"] for hit in hits]
    assert all(item["matched_text"] == "quick" for item in results)
    assert all(item["rects"] for item in results)
