"""Semantic question-answering API tests with an injected fake provider.

The embedding double is deterministic and cross-lingual only by fixture
design. These tests verify traceability, error contracts and the absence of a
literal fallback; they do not measure real embedding quality.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from semantic_fakes import (
    VECTOR_DIMENSION,
    FakeEmbeddingProvider,
    fake_identity,
    keyword_vector,
)

from rag_anchor.infra.embedding import (
    EmbeddingModelIdentity,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
)
from rag_anchor.infra.llm import LLMMessage

DOCUMENTS_URL = "/api/v1/documents"
SEMANTIC_INDEX_URL = DOCUMENTS_URL + "/{document_id}/semantic-index"
QA_URL = DOCUMENTS_URL + "/{document_id}/qa"
EVIDENCE_URL = DOCUMENTS_URL + "/{document_id}/evidence/locate"

STAGES = "The project has three stages: ingestion, retrieval, and evidence verification."
PHOTO = "The photo is stored separately."
CHINESE_QUESTION = "这个项目有哪些阶段？"  # noqa: RUF001

QA_FIELDS = {
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


@dataclass
class RecordingLLMProvider:
    output: str | Exception
    calls: list[list[LLMMessage]] = field(default_factory=list)

    def complete(self, messages) -> str:
        self.calls.append(list(messages))
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


@dataclass
class ExplodingEmbeddingProvider:
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"
    calls: int = 0

    def resolve_identity(self) -> EmbeddingModelIdentity:
        raise AssertionError("literal retrieval must not resolve an embedding identity")

    def embed(self, texts):
        self.calls += 1
        raise AssertionError("literal retrieval must not call the embedding provider")


@dataclass
class BadDimensionProvider:
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts):
        return [[1.0] * 8 for _ in texts[:-1]] + [[1.0] * 4]


@dataclass
class SwitchableProvider:
    """Succeeds for indexing, then fails only for the query embedding."""

    error: Exception | None = None
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts):
        if self.error is not None:
            raise self.error
        return [keyword_vector(text) for text in texts]


@dataclass
class MutableDigestProvider:
    """Same tag and version label, but the server-reported digest can change."""

    digest: str = "digest-1"
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return EmbeddingModelIdentity(name=self.model_name, digest=self.digest)

    def embed(self, texts):
        return [keyword_vector(text) for text in texts]


@dataclass
class DeletingProvider:
    """Deletes the document while the build is embedding, simulating a race."""

    delete: Callable[[], bool] | None = None
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts):
        assert self.delete is not None
        self.delete()
        return [keyword_vector(text) for text in texts]


@pytest.fixture
def semantic_pdf(pdf_factory) -> Path:
    return pdf_factory(
        "semantic.pdf",
        [
            [(100.0, 200.0, STAGES), (100.0, 240.0, PHOTO)],
        ],
    )


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


def _build_semantic(client: TestClient, document_id: str):
    return client.post(SEMANTIC_INDEX_URL.format(document_id=document_id))


def _ready(
    api,
    path: Path,
) -> str:
    document_id = _upload(api.client, path)
    _index(api.client, document_id)
    response = _build_semantic(api.client, document_id)
    assert response.status_code == 200, response.text
    return document_id


def _ask(client: TestClient, document_id: str, payload: dict[str, Any]):
    return client.post(QA_URL.format(document_id=document_id), json=payload)


def _locate(client: TestClient, document_id: str, items: list[dict[str, object]]):
    return client.post(EVIDENCE_URL.format(document_id=document_id), json={"items": items})


def _pages(api, document_id: str) -> dict[int, str]:
    document_uuid = UUID(document_id)
    return {
        page.page_index: page.text
        for page in api.app.state.document_service.repository.get_document_page_texts(document_uuid)
    }


def test_semantic_build_endpoint_returns_index_provenance(api_factory, semantic_pdf: Path) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _upload(api.client, semantic_pdf)
    _index(api.client, document_id)

    response = _build_semantic(api.client, document_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["document_id"] == document_id
    assert body["model_name"] == "fake-bge-m3"
    assert body["model_version"] == "v1"
    assert body["model_digest"] == "fake-digest:v1"
    assert body["index_version"] == 1
    assert body["candidate_version"] == 1
    assert body["dimension"] == VECTOR_DIMENSION
    assert body["normalization"] == "l2"
    assert body["source_fingerprint"]
    assert body["candidate_count"] == 2
    assert body["built_at"]


def test_build_requires_an_existing_document(api_factory) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())

    response = _build_semantic(api.client, "00000000-0000-0000-0000-000000000000")

    assert response.status_code == 404, response.text


def test_missing_document_is_404_before_disabled_503(api) -> None:
    missing = "00000000-0000-0000-0000-000000000000"

    build = _build_semantic(api.client, missing)
    ask = _ask(api.client, missing, {"question": "anything", "retrieval_mode": "semantic"})

    assert build.status_code == 404, build.text
    assert ask.status_code == 404, ask.text


def test_document_deleted_during_build_returns_404(api_factory, semantic_pdf: Path) -> None:
    provider = DeletingProvider()
    api = api_factory(embedding_provider=provider)
    document_id = _upload(api.client, semantic_pdf)
    _index(api.client, document_id)
    provider.delete = lambda: api.app.state.document_service.repository.delete(UUID(document_id))

    response = _build_semantic(api.client, document_id)

    assert response.status_code == 404, response.text


def test_build_requires_the_text_index(api_factory, semantic_pdf: Path) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _upload(api.client, semantic_pdf)

    response = _build_semantic(api.client, document_id)

    assert response.status_code == 409, response.text
    assert "text index" in response.json()["detail"].lower()


def test_disabled_embedding_provider_returns_503(api, semantic_pdf: Path) -> None:
    document_id = _upload(api.client, semantic_pdf)
    _index(api.client, document_id)

    build = _build_semantic(api.client, document_id)
    ask = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert build.status_code == 503, build.text
    assert ask.status_code == 503, ask.text


def test_invalid_embedding_dimensions_return_502(api_factory, semantic_pdf: Path) -> None:
    api = api_factory(embedding_provider=BadDimensionProvider())
    document_id = _upload(api.client, semantic_pdf)
    _index(api.client, document_id)

    response = _build_semantic(api.client, document_id)

    assert response.status_code == 502, response.text


def test_semantic_question_finds_english_evidence_for_a_chinese_question(
    api_factory,
    semantic_pdf: Path,
) -> None:
    fake_llm = RecordingLLMProvider('{"answer": "unused", "citation_ids": ["E1"]}')
    api = api_factory(
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=fake_llm,
    )
    document_id = _ready(api, semantic_pdf)

    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == QA_FIELDS
    assert body["retrieval_mode"] == "semantic"
    assert body["retrieval_query"] == CHINESE_QUESTION
    assert body["answer_mode"] == "evidence_only"
    assert body["status"] == "evidence_found"
    assert "语义" in body["answer"]
    assert "不代表事实支持" in body["answer"]
    assert fake_llm.calls == []

    pages = _pages(api, document_id)
    assert body["citations"]
    citation = body["citations"][0]
    assert citation["retrieval_kind"] == "semantic"
    assert citation["support_kind"] == "semantic_candidate"
    assert citation["score"] is not None
    assert citation["hit_text"] == citation["support_text"]
    assert citation["match_start"] == citation["support_start"]
    assert citation["match_end"] == citation["support_end"]
    assert (
        pages[citation["page_index"]][citation["support_start"] : citation["support_end"]]
        == citation["support_text"]
    )
    assert citation["support_text"] in citation["chunk_text"]
    assert citation["support_text"] != citation["chunk_text"]


def test_semantic_citation_locates_the_exact_pdf_span(api_factory, semantic_pdf: Path) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _ready(api, semantic_pdf)
    citations = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    ).json()["citations"]

    item = {
        "page_index": citations[0]["page_index"],
        "match_start": citations[0]["support_start"],
        "match_end": citations[0]["support_end"],
        "hit_text": citations[0]["support_text"],
        "chunk_id": citations[0]["chunk_id"],
    }
    response = _locate(api.client, document_id, [item])

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["matched_text"] == citations[0]["support_text"]
    assert result["char_count"] > 0
    assert result["rects"]
    assert result["chunk_id"] == citations[0]["chunk_id"]


def test_literal_mode_is_still_the_default_and_does_not_fall_back(
    api_factory,
    semantic_pdf: Path,
) -> None:
    provider = ExplodingEmbeddingProvider()
    api = api_factory(embedding_provider=provider)
    document_id = _upload(api.client, semantic_pdf)
    _index(api.client, document_id)
    provider.calls = 0

    default = _ask(api.client, document_id, {"question": "project"})
    chinese = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "literal"},
    )

    assert default.status_code == 200, default.text
    assert default.json()["retrieval_mode"] == "literal"
    assert default.json()["status"] == "evidence_found"
    assert default.json()["citations"][0]["retrieval_kind"] == "literal"
    assert default.json()["citations"][0]["score"] is None

    assert chinese.status_code == 200, chinese.text
    assert chinese.json()["retrieval_mode"] == "literal"
    assert chinese.json()["status"] == "insufficient_evidence"
    assert chinese.json()["citations"] == []
    assert provider.calls == 0


def test_semantic_mode_never_falls_back_to_literal_matching(
    api_factory,
    semantic_pdf: Path,
) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _ready(api, semantic_pdf)

    response = _ask(
        api.client,
        document_id,
        {
            "question": "quantum chromodynamics",
            "retrieval_query": "stages",
            "retrieval_mode": "semantic",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retrieval_mode"] == "semantic"
    assert body["retrieval_query"] == "quantum chromodynamics"
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []
    assert "相似度" in body["answer"]


def test_semantic_index_missing_returns_409(api_factory, semantic_pdf: Path) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _upload(api.client, semantic_pdf)
    _index(api.client, document_id)

    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert response.status_code == 409, response.text
    assert "semantic index" in response.json()["detail"].lower()


def test_rebuilding_the_text_index_invalidates_semantic_qa(
    api_factory,
    semantic_pdf: Path,
) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _ready(api, semantic_pdf)

    _index(api.client, document_id)
    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert response.status_code == 409, response.text


def test_changed_model_version_is_reported_as_stale(
    api_factory,
    semantic_pdf: Path,
) -> None:
    provider = FakeEmbeddingProvider()
    api = api_factory(embedding_provider=provider)
    document_id = _ready(api, semantic_pdf)

    provider.model_version = "v2"
    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert response.status_code == 409, response.text
    assert "different embedding model" in response.json()["detail"]


def test_changed_model_digest_is_reported_as_stale(
    api_factory,
    semantic_pdf: Path,
) -> None:
    provider = MutableDigestProvider()
    api = api_factory(embedding_provider=provider)
    document_id = _ready(api, semantic_pdf)

    provider.digest = "digest-2"
    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert response.status_code == 409, response.text
    assert "identity changed" in response.json()["detail"]


def test_semantic_grounded_generation_sends_the_ranked_candidate(
    api_factory,
    semantic_pdf: Path,
) -> None:
    fake_llm = RecordingLLMProvider(
        json.dumps({"answer": "项目分为三个阶段。", "citation_ids": ["E1"]}, ensure_ascii=False)
    )
    api = api_factory(
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=fake_llm,
    )
    document_id = _ready(api, semantic_pdf)

    response = _ask(
        api.client,
        document_id,
        {
            "question": CHINESE_QUESTION,
            "retrieval_mode": "semantic",
            "answer_mode": "grounded_generation",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retrieval_mode"] == "semantic"
    assert body["answer_mode"] == "grounded_generation"
    assert body["status"] == "answered"
    assert body["answer"] == "项目分为三个阶段。"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["citation_id"] == "E1"

    assert len(fake_llm.calls) == 1
    payload = json.loads(fake_llm.calls[0][-1].content)
    evidence = payload["evidence"]
    assert evidence[0]["id"] == "E1"
    assert evidence[0]["text"] == body["citations"][0]["support_text"]
    assert body["citations"][0]["support_text"] != body["citations"][0]["chunk_text"]


def test_semantic_grounded_no_evidence_does_not_call_the_model(
    api_factory,
    semantic_pdf: Path,
) -> None:
    fake_llm = RecordingLLMProvider('{"answer": "forged", "citation_ids": ["E1"]}')
    api = api_factory(
        embedding_provider=FakeEmbeddingProvider(),
        llm_provider=fake_llm,
    )
    document_id = _ready(api, semantic_pdf)

    response = _ask(
        api.client,
        document_id,
        {
            "question": "quantum chromodynamics",
            "retrieval_mode": "semantic",
            "answer_mode": "grounded_generation",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []
    assert "没有" in body["answer"]
    assert fake_llm.calls == []


def test_invalid_retrieval_mode_is_rejected(api_factory, semantic_pdf: Path) -> None:
    api = api_factory(embedding_provider=FakeEmbeddingProvider())
    document_id = _ready(api, semantic_pdf)

    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "hybrid"},
    )

    assert response.status_code == 422, response.text


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (EmbeddingServiceError("service down"), 502),
        (EmbeddingTimeoutError("timed out"), 504),
    ],
)
def test_query_time_embedding_failure_is_reported(
    api_factory,
    semantic_pdf: Path,
    error: Exception,
    status_code: int,
) -> None:
    provider = SwitchableProvider()
    api = api_factory(embedding_provider=provider)
    document_id = _ready(api, semantic_pdf)

    provider.error = error
    response = _ask(
        api.client,
        document_id,
        {"question": CHINESE_QUESTION, "retrieval_mode": "semantic"},
    )

    assert response.status_code == status_code, response.text
    assert response.json()["detail"]
