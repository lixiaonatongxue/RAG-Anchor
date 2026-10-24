"""Offline tests for the injectable Ollama embedding provider."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from rag_anchor.config import EmbeddingSettings
from rag_anchor.infra.embedding import (
    DisabledEmbeddingProvider,
    EmbeddingDisabledError,
    EmbeddingResponseError,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
    OllamaEmbeddingProvider,
    build_embedding_provider,
)
from rag_anchor.infra.llm import (
    HTTPResponse,
    HTTPTimeoutError,
    HTTPTransportError,
)


@dataclass
class FakeTransport:
    response: HTTPResponse | None = None
    error: Exception | None = None
    get_response: HTTPResponse | None = None
    get_error: Exception | None = None
    calls: list[dict[str, object]] = field(default_factory=list)
    get_calls: list[dict[str, object]] = field(default_factory=list)

    def post(
        self,
        url: str,
        *,
        headers,
        body: bytes,
        timeout: float,
    ) -> HTTPResponse:
        self.calls.append({"url": url, "headers": dict(headers), "body": body, "timeout": timeout})
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def get(self, url: str, *, headers, timeout: float) -> HTTPResponse:
        self.get_calls.append({"url": url, "headers": dict(headers), "timeout": timeout})
        if self.get_error is not None:
            raise self.get_error
        assert self.get_response is not None
        return self.get_response


def _provider(transport: FakeTransport) -> OllamaEmbeddingProvider:
    return OllamaEmbeddingProvider(
        base_url="http://127.0.0.1:11434",
        model="bge-m3",
        model_version="digest-1",
        api_key="secret",
        timeout_seconds=12.5,
        transport=transport,
    )


def _response(payload: object, status: int = 200) -> HTTPResponse:
    return HTTPResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def _tags(models: object, status: int = 200) -> HTTPResponse:
    return _response({"models": models}, status=status)


def test_identity_reads_the_matching_model_digest() -> None:
    transport = FakeTransport(
        get_response=_tags(
            [
                {"name": "other:latest", "digest": "sha256:other"},
                {
                    "name": "bge-m3:latest",
                    "model": "bge-m3:latest",
                    "digest": "sha256:model",
                },
            ]
        )
    )
    provider = _provider(transport)

    identity = provider.resolve_identity()

    assert identity.name == "bge-m3:latest"
    assert identity.digest == "sha256:model"
    call = transport.get_calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/tags"
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["timeout"] == 12.5


def test_identity_is_resolved_on_every_call_and_not_cached() -> None:
    transport = FakeTransport(
        get_response=_tags([{"name": "bge-m3:latest", "digest": "sha256:model"}])
    )
    provider = _provider(transport)

    first = provider.resolve_identity()
    second = provider.resolve_identity()

    assert first == second
    assert len(transport.get_calls) == 2


def test_identity_requires_the_configured_model_to_be_reported() -> None:
    transport = FakeTransport(get_response=_tags([{"name": "qwen3:8b", "digest": "sha256:other"}]))

    with pytest.raises(EmbeddingResponseError):
        _provider(transport).resolve_identity()


def test_identity_requires_a_digest() -> None:
    transport = FakeTransport(get_response=_tags([{"name": "bge-m3:latest"}]))

    with pytest.raises(EmbeddingResponseError):
        _provider(transport).resolve_identity()


def test_identity_rejects_ambiguous_matching_digests() -> None:
    transport = FakeTransport(
        get_response=_tags(
            [
                {"name": "bge-m3", "digest": "sha256:one"},
                {"name": "bge-m3:latest", "digest": "sha256:two"},
            ]
        )
    )

    with pytest.raises(EmbeddingResponseError):
        _provider(transport).resolve_identity()


@pytest.mark.parametrize(
    "get_response",
    [
        HTTPResponse(status=200, body=b"not json"),
        HTTPResponse(status=200, body=json.dumps([]).encode()),
        HTTPResponse(status=200, body=json.dumps({"models": "nope"}).encode()),
    ],
)
def test_identity_rejects_malformed_model_lists(get_response: HTTPResponse) -> None:
    transport = FakeTransport(get_response=get_response)

    with pytest.raises(EmbeddingResponseError):
        _provider(transport).resolve_identity()


def test_identity_timeout_is_reported_separately() -> None:
    transport = FakeTransport(get_error=HTTPTimeoutError("timed out"))

    with pytest.raises(EmbeddingTimeoutError):
        _provider(transport).resolve_identity()


@pytest.mark.parametrize(
    "get_response",
    [HTTPResponse(status=500, body=b""), HTTPResponse(status=404, body=b"")],
)
def test_identity_http_error_is_a_service_error(get_response: HTTPResponse) -> None:
    transport = FakeTransport(get_response=get_response)

    with pytest.raises(EmbeddingServiceError):
        _provider(transport).resolve_identity()


def test_identity_transport_failure_is_a_service_error() -> None:
    transport = FakeTransport(get_error=HTTPTransportError("failed"))

    with pytest.raises(EmbeddingServiceError):
        _provider(transport).resolve_identity()


def test_provider_posts_to_ollama_embed_and_returns_vectors() -> None:
    transport = FakeTransport(response=_response({"embeddings": [[3.0, 4.0], [0.0, 1.0]]}))
    provider = _provider(transport)

    vectors = provider.embed(["first", "second"])

    assert vectors == [[3.0, 4.0], [0.0, 1.0]]
    assert provider.model_name == "bge-m3"
    assert provider.model_version == "digest-1"
    call = transport.calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/embed"
    assert call["timeout"] == 12.5
    assert call["headers"]["Authorization"] == "Bearer secret"
    payload = json.loads(call["body"])
    assert payload == {"model": "bge-m3", "input": ["first", "second"]}


def test_empty_input_never_calls_the_transport() -> None:
    transport = FakeTransport(error=AssertionError("must not be called"))
    assert _provider(transport).embed([]) == []
    assert transport.calls == []


def test_whitespace_input_is_rejected_before_the_request() -> None:
    transport = FakeTransport()
    with pytest.raises(ValueError):
        _provider(transport).embed(["   "])
    assert transport.calls == []


def test_http_error_is_a_service_error() -> None:
    transport = FakeTransport(response=HTTPResponse(status=404, body=b""))
    with pytest.raises(EmbeddingServiceError):
        _provider(transport).embed(["query"])


def test_timeout_is_reported_separately() -> None:
    transport = FakeTransport(error=HTTPTimeoutError("timed out"))
    with pytest.raises(EmbeddingTimeoutError):
        _provider(transport).embed(["query"])


def test_transport_failure_is_a_service_error() -> None:
    transport = FakeTransport(error=HTTPTransportError("failed"))
    with pytest.raises(EmbeddingServiceError):
        _provider(transport).embed(["query"])


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        json.dumps({"embeddings": "nope"}).encode(),
        json.dumps({"embeddings": []}).encode(),
        json.dumps({"embeddings": [[1.0], [2.0]]}).encode(),
        json.dumps({"embeddings": ["vector"]}).encode(),
        json.dumps({"embeddings": [[]]}).encode(),
        json.dumps({"embeddings": [[True]]}).encode(),
        json.dumps({"embeddings": [["1.0"]]}).encode(),
        json.dumps({"embeddings": [[float("nan")]]}).encode(),
        json.dumps({"embeddings": [[float("inf")]]}).encode(),
        json.dumps({"embeddings": [[1.0, 2.0], [3.0]]}).encode(),
    ],
)
def test_invalid_embedding_envelopes_are_rejected(body: bytes) -> None:
    transport = FakeTransport(response=HTTPResponse(status=200, body=body))
    with pytest.raises(EmbeddingResponseError):
        _provider(transport).embed(["query"])


def test_build_provider_is_disabled_by_default() -> None:
    provider = build_embedding_provider(EmbeddingSettings())
    assert isinstance(provider, DisabledEmbeddingProvider)
    with pytest.raises(EmbeddingDisabledError):
        provider.embed(["query"])


def test_build_provider_requires_url_and_model() -> None:
    provider = build_embedding_provider(
        EmbeddingSettings(enabled=True, base_url="http://127.0.0.1:11434")
    )
    assert isinstance(provider, DisabledEmbeddingProvider)


def test_build_provider_creates_ollama_provider_when_enabled() -> None:
    provider = build_embedding_provider(
        EmbeddingSettings(
            enabled=True,
            base_url="http://127.0.0.1:11434",
            model="bge-m3",
            model_version="v1",
        )
    )
    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model_name == "bge-m3"
    assert provider.model_version == "v1"


def test_embedding_settings_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        EmbeddingSettings(base_url="ftp://example.com")
    with pytest.raises(ValueError):
        EmbeddingSettings(timeout_seconds=0)
    with pytest.raises(ValueError):
        EmbeddingSettings(batch_size=0)
    with pytest.raises(ValueError):
        EmbeddingSettings(min_score=1.5)


def test_embedding_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MODEL", "bge-m3")
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MODEL_VERSION", "digest-abc")
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("RAG_ANCHOR_EMBEDDING_MIN_SCORE", "0.42")

    settings = EmbeddingSettings.from_env()

    assert settings.enabled is True
    assert settings.configured is True
    assert settings.resolved_model_version == "digest-abc"
    assert settings.batch_size == 4
    assert settings.min_score == pytest.approx(0.42)
