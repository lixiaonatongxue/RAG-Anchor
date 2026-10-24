"""Evidence-only question answering API tests.

The endpoint must never invent an answer or a citation. Every assertion below
ties the response back to the persisted chunk index, and citation spans are
fed into the existing evidence location endpoint exactly as the frontend does.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from rag_anchor.services.qa_service import (
    QuestionAnswerService,
    QuestionCitationScopeError,
)
from rag_anchor.services.retrieval_service import RetrievalMatch, RetrievalResult

DOCUMENTS_URL = "/api/v1/documents"
RETRIEVAL_URL = "/api/v1/retrieval/search"
EVIDENCE_URL = "/api/v1/documents/{document_id}/evidence/locate"
QA_URL = "/api/v1/documents/{document_id}/qa"

RESPONSE_FIELDS = {
    "document_id",
    "question",
    "retrieval_query",
    "retrieval_mode",
    "match_case",
    "answer_mode",
    "status",
    "answer",
    "total_matches",
    "citations",
}

CITATION_FIELDS = {
    "citation_id",
    "document_id",
    "chunk_id",
    "page_index",
    "page_number",
    "chunk_index",
    "match_start",
    "match_end",
    "hit_text",
    "chunk_text",
    "extraction_version",
    "retrieval_kind",
    "score",
    "support_text",
    "support_start",
    "support_end",
    "support_kind",
}


def _upload(client: TestClient, path: Path, filename: str = "document.pdf") -> str:
    response = client.post(
        DOCUMENTS_URL,
        files={"file": (filename, path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


def _index(client: TestClient, document_id: str) -> dict:
    response = client.post(f"{DOCUMENTS_URL}/{document_id}/index")
    assert response.status_code == 200, response.text
    return response.json()


def _ask(client: TestClient, document_id: str, payload: dict[str, object]):
    return client.post(QA_URL.format(document_id=document_id), json=payload)


def _retrieval_hits(
    client: TestClient,
    document_id: str,
    query: str,
    match_case: bool = True,
) -> list[dict]:
    response = client.post(
        RETRIEVAL_URL,
        json={
            "query": query,
            "document_id": document_id,
            "match_case": match_case,
            "limit": 20,
            "offset": 0,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["results"]


def _citation_item(citation: dict) -> dict[str, object]:
    """Build the evidence/locate item exactly as the frontend does.

    The support span is what a citation click locates; the original retrieval
    hit fields are never used for geometry.
    """
    return {
        "page_index": citation["page_index"],
        "match_start": citation["support_start"],
        "match_end": citation["support_end"],
        "hit_text": citation["support_text"],
        "chunk_id": citation["chunk_id"],
    }


def _page_texts(api, document_id: str) -> dict[int, str]:
    repository = api.app.state.document_service.repository
    return {
        page.page_index: page.text for page in repository.get_document_page_texts(UUID(document_id))
    }


def _indexed_document(api, path: Path, filename: str = "document.pdf") -> str:
    document_id = _upload(api.client, path, filename)
    _index(api.client, document_id)
    return document_id


def test_evidence_only_answer_returns_traceable_citations(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    response = _ask(
        api.client,
        document_id,
        {
            "question": "What does the document say about the animal?",
            "retrieval_query": "quick",
            "match_case": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["document_id"] == document_id
    assert body["question"] == "What does the document say about the animal?"
    assert body["retrieval_query"] == "quick"
    assert body["retrieval_mode"] == "literal"
    assert body["match_case"] is True
    assert body["answer_mode"] == "evidence_only"
    assert body["status"] == "evidence_found"
    assert "证据展示模式" in body["answer"]
    assert "未接入语言模型" in body["answer"]
    assert body["total_matches"] == 3
    assert len(body["citations"]) == 3

    retrieval = _retrieval_hits(api.client, document_id, "quick", match_case=True)
    expected = [
        (
            hit["page_index"],
            hit["match_start"],
            hit["match_end"],
            hit["hit_text"],
            hit["chunk_id"],
            hit["chunk_index"],
            hit["chunk_text"],
        )
        for hit in retrieval[:3]
    ]
    actual = [
        (
            citation["page_index"],
            citation["match_start"],
            citation["match_end"],
            citation["hit_text"],
            citation["chunk_id"],
            citation["chunk_index"],
            citation["chunk_text"],
        )
        for citation in body["citations"]
    ]
    assert actual == expected

    page_texts = _page_texts(api, document_id)
    for index, citation in enumerate(body["citations"], start=1):
        assert set(citation) == CITATION_FIELDS
        assert citation["citation_id"] == f"E{index}"
        assert citation["document_id"] == document_id
        assert citation["page_index"] == 0
        assert citation["page_number"] == 1
        assert citation["match_start"] < citation["match_end"]
        assert len(citation["hit_text"]) == citation["match_end"] - citation["match_start"]
        assert citation["extraction_version"]
        assert citation["support_kind"] in {"sentence", "retrieval_match"}
        assert citation["support_start"] <= citation["match_start"]
        assert citation["match_end"] <= citation["support_end"]
        page_text = page_texts[citation["page_index"]]
        assert (
            page_text[citation["support_start"] : citation["support_end"]]
            == citation["support_text"]
        )
        assert (
            citation["support_text"][
                citation["match_start"] - citation["support_start"] : citation["match_end"]
                - citation["support_start"]
            ]
            == citation["hit_text"]
        )
        if citation["support_kind"] == "retrieval_match":
            assert citation["support_start"] == citation["match_start"]
            assert citation["support_end"] == citation["match_end"]
            assert citation["support_text"] == citation["hit_text"]


def test_retrieval_query_defaults_to_the_question(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    response = _ask(api.client, document_id, {"question": "fox"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "evidence_found"
    assert body["retrieval_query"] == "fox"
    assert body["match_case"] is False
    assert body["total_matches"] == 1
    assert body["citations"][0]["hit_text"] == "fox"


def test_no_evidence_returns_explicit_refusal(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    response = _ask(
        api.client,
        document_id,
        {
            "question": "What is the main contribution?",
            "retrieval_query": "definitely-not-present",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["total_matches"] == 0
    assert body["citations"] == []
    assert "无法依据该文档回答" in body["answer"]
    assert "未接入语言模型" in body["answer"]


def test_question_is_scoped_to_one_document(api, single_page_pdf: Path) -> None:
    first = _indexed_document(api, single_page_pdf, "first.pdf")
    second = _indexed_document(api, single_page_pdf, "second.pdf")

    first_answer = _ask(api.client, first, {"question": "quick"})
    second_answer = _ask(api.client, second, {"question": "quick"})

    assert first_answer.status_code == 200, first_answer.text
    assert second_answer.status_code == 200, second_answer.text
    assert {citation["document_id"] for citation in first_answer.json()["citations"]} == {first}
    assert {citation["document_id"] for citation in second_answer.json()["citations"]} == {second}


def test_unindexed_document_returns_409(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)

    response = _ask(api.client, document_id, {"question": "quick"})

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "Document has not been indexed"


def test_unknown_document_returns_404(api) -> None:
    response = _ask(
        api.client,
        "00000000-0000-0000-0000-000000000000",
        {"question": "quick"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_deleted_document_returns_404(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    assert api.client.delete(f"{DOCUMENTS_URL}/{document_id}").status_code == 204

    response = _ask(api.client, document_id, {"question": "quick"})

    assert response.status_code == 404, response.text


def test_max_citations_caps_the_returned_list_but_not_total(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    response = _ask(
        api.client,
        document_id,
        {"question": "quick", "match_case": True, "max_citations": 2},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_matches"] == 3
    assert len(body["citations"]) == 2
    assert [citation["citation_id"] for citation in body["citations"]] == ["E1", "E2"]
    assert "以下返回 2 条引用" in body["answer"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": None},
        {"question": "ok", "retrieval_query": ""},
        {"question": "ok", "retrieval_query": "   "},
        {"question": "ok", "max_citations": 0},
        {"question": "ok", "max_citations": 21},
        {"question": "x" * 2001},
    ],
)
def test_invalid_payload_is_rejected(api, payload: dict[str, object]) -> None:
    response = _ask(
        api.client,
        "00000000-0000-0000-0000-000000000000",
        payload,
    )

    assert response.status_code == 422, response.text


def test_citations_locate_to_distinct_real_occurrences(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    body = _ask(
        api.client,
        document_id,
        {"question": "quick", "match_case": True},
    ).json()

    locate = api.client.post(
        EVIDENCE_URL.format(document_id=document_id),
        json={"items": [_citation_item(citation) for citation in body["citations"]]},
    )

    assert locate.status_code == 200, locate.text
    results = locate.json()["results"]
    assert [item["match_start"] for item in results] == [
        citation["support_start"] for citation in body["citations"]
    ]
    assert [item["hit_text"] for item in results] == [
        citation["support_text"] for citation in body["citations"]
    ]
    assert [item["matched_text"] for item in results] == [
        citation["support_text"] for citation in body["citations"]
    ]
    assert all(item["rects"] for item in results)
    rect_sets = {tuple(sorted(rect.items())) for item in results for rect in item["rects"]}
    assert len(rect_sets) == len(body["citations"])


def test_qa_does_not_change_index_rows(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    repository = api.app.state.document_service.repository
    document_uuid = UUID(document_id)
    chunks_before = repository.get_document_chunks(document_uuid)
    pages_before = repository.get_document_page_texts(document_uuid)
    metadata_before = repository.get_document_index_metadata(document_uuid)

    _ask(api.client, document_id, {"question": "quick"})
    _ask(api.client, document_id, {"question": "not-present"})

    assert repository.get_document_chunks(document_uuid) == chunks_before
    assert repository.get_document_page_texts(document_uuid) == pages_before
    assert repository.get_document_index_metadata(document_uuid) == metadata_before


def test_keyword_hit_expands_to_the_complete_sentence(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)

    body = _ask(
        api.client,
        document_id,
        {"question": "stages", "retrieval_query": "quick", "max_citations": 1},
    ).json()

    citation = body["citations"][0]
    assert citation["hit_text"] == "quick"
    assert citation["support_kind"] == "sentence"
    assert citation["support_text"] == "The quick brown fox jumps over the lazy dog."
    assert citation["support_start"] < citation["match_start"]
    assert citation["match_end"] < citation["support_end"]
    assert citation["support_start"] >= 0

    locate = api.client.post(
        EVIDENCE_URL.format(document_id=document_id),
        json={"items": [_citation_item(citation)]},
    )
    assert locate.status_code == 200, locate.text
    result = locate.json()["results"][0]
    assert result["matched_text"] == citation["support_text"]
    assert result["match_start"] == citation["support_start"]
    assert result["match_end"] == citation["support_end"]
    # One line of text yields exactly the real PDFium rectangle for that line;
    # the support span never adds synthetic rectangles.
    assert len(result["rects"]) == 1
    assert result["bbox"] is not None


def test_cross_line_sentence_falls_back_to_the_retrieval_hit(
    api,
    pdf_factory,
) -> None:
    path = pdf_factory(
        "crossline.pdf",
        [
            [
                (100.0, 300.0, "The project has three"),
                (100.0, 280.0, "stages: ingestion, retrieval, and evidence verification."),
            ]
        ],
    )
    document_id = _indexed_document(api, path)

    body = _ask(
        api.client,
        document_id,
        {"question": "stages", "retrieval_query": "stages"},
    ).json()

    citation = body["citations"][0]
    assert citation["hit_text"] == "stages"
    assert citation["support_kind"] == "retrieval_match"
    assert citation["support_text"] == citation["hit_text"]
    assert citation["support_start"] == citation["match_start"]
    assert citation["support_end"] == citation["match_end"]
    page_text = _page_texts(api, document_id)[citation["page_index"]]
    assert "\r\n" in page_text
    assert page_text[citation["support_start"] : citation["support_end"]] == "stages"

    locate = api.client.post(
        EVIDENCE_URL.format(document_id=document_id),
        json={"items": [_citation_item(citation)]},
    )
    assert locate.status_code == 200, locate.text
    assert locate.json()["results"][0]["matched_text"] == "stages"


def test_repeated_identical_sentence_gets_distinct_exact_support_spans(
    api,
    pdf_factory,
) -> None:
    sentence = "The project has three stages."
    path = pdf_factory(
        "repeat-sentence.pdf",
        [[(100.0, 300.0, sentence), (100.0, 280.0, sentence)]],
    )
    document_id = _indexed_document(api, path)

    body = _ask(
        api.client,
        document_id,
        {"question": "stages", "retrieval_query": "stages"},
    ).json()

    citations = body["citations"]
    assert len(citations) == 2
    assert [citation["support_kind"] for citation in citations] == ["sentence", "sentence"]
    assert [citation["support_text"] for citation in citations] == [sentence, sentence]
    assert citations[0]["support_start"] != citations[1]["support_start"]

    page_text = _page_texts(api, document_id)[0]
    for citation in citations:
        assert page_text[citation["support_start"] : citation["support_end"]] == sentence

    locate = api.client.post(
        EVIDENCE_URL.format(document_id=document_id),
        json={"items": [_citation_item(citation) for citation in citations]},
    )
    assert locate.status_code == 200, locate.text
    results = locate.json()["results"]
    assert [result["matched_text"] for result in results] == [sentence, sentence]
    assert results[0]["bbox"] != results[1]["bbox"]


def test_sentence_cut_by_chunk_boundary_falls_back(
    api,
    pdf_factory,
) -> None:
    # 810 characters: the hit sits in the first 800-character chunk while its
    # terminal period is only in the next chunk. The chunk is never extended.
    text = ("A" * 780) + " stages " + ("B" * 20) + "."
    path = pdf_factory("chunk-boundary.pdf", [[(100.0, 300.0, text)]])
    document_id = _indexed_document(api, path)

    body = _ask(
        api.client,
        document_id,
        {"question": "stages", "retrieval_query": "stages"},
    ).json()

    citation = body["citations"][0]
    assert citation["chunk_index"] == 0
    assert citation["support_kind"] == "retrieval_match"
    assert citation["support_text"] == "stages"
    assert citation["support_start"] == citation["match_start"]
    assert citation["support_end"] == citation["match_end"]


def test_cjk_punctuation_selects_the_complete_sentence(api, pdf_factory) -> None:
    font_name = "STSong-Light"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    sentence = "项目包含三个阶段：采集、检索和证据校验。预算尚未确定。"  # noqa: RUF001
    path = pdf_factory(
        "cjk-support.pdf",
        [[(100.0, 300.0, sentence)]],
        font_name=font_name,
    )
    document_id = _indexed_document(api, path)

    body = _ask(
        api.client,
        document_id,
        {"question": "证据校验", "retrieval_query": "证据校验"},
    ).json()

    citation = body["citations"][0]
    assert citation["support_kind"] == "sentence"
    assert citation["support_text"] == (
        "项目包含三个阶段：采集、检索和证据校验。"  # noqa: RUF001
    )
    page_text = _page_texts(api, document_id)[citation["page_index"]]
    assert (
        page_text[citation["support_start"] : citation["support_end"]] == citation["support_text"]
    )

    locate = api.client.post(
        EVIDENCE_URL.format(document_id=document_id),
        json={"items": [_citation_item(citation)]},
    )
    assert locate.status_code == 200, locate.text
    assert locate.json()["results"][0]["matched_text"] == citation["support_text"]


class _ForeignDocumentRetrieval:
    """Test double whose result violates the document scope."""

    def __init__(self, foreign_document_id: UUID) -> None:
        self._foreign_document_id = foreign_document_id

    def search_chunks(self, query: str, **kwargs: object) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            match_case=bool(kwargs["match_case"]),
            document_id=kwargs["document_id"],  # type: ignore[arg-type]
            total=1,
            limit=int(kwargs["limit"]),  # type: ignore[arg-type]
            offset=0,
            matches=[
                RetrievalMatch(
                    document_id=self._foreign_document_id,
                    chunk_id="chk_foreign",
                    page_index=0,
                    page_number=1,
                    chunk_index=0,
                    start_offset=0,
                    end_offset=5,
                    match_start=0,
                    match_end=5,
                    hit_text="quick",
                    chunk_text="quick",
                    extraction_version="v1",
                )
            ],
        )


def test_foreign_document_citation_is_rejected(api, single_page_pdf: Path) -> None:
    document_id = _indexed_document(api, single_page_pdf)
    foreign_document_id = _upload(api.client, single_page_pdf, "foreign.pdf")
    service = QuestionAnswerService(
        api.app.state.document_service,
        _ForeignDocumentRetrieval(UUID(foreign_document_id)),
    )

    with pytest.raises(QuestionCitationScopeError):
        service.answer(UUID(document_id), "quick")
