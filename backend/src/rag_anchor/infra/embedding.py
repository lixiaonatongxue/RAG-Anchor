"""Injectable embedding provider for semantic retrieval.

The default implementation talks to the official Ollama ``POST /api/embed``
endpoint and reads model identities from ``GET /api/tags``. No request is made
at import time, application startup or while settings are merely configured. A
provider is only called when a semantic index is explicitly built or a semantic
question is explicitly asked.

The provider validates the response envelope strictly: every input text must
receive exactly one non-empty vector, all values must be finite, and all
vectors must share one dimension. Raw vectors are returned unchanged; L2
normalization is applied by the semantic index service and recorded in the
index metadata.

Identity is resolved from the server-reported model list, not from the mutable
tag or the operator-supplied version label. A digest is required: if the model
is absent, has no digest or is reported ambiguously, identity resolution fails
and no index build or query proceeds.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_anchor.config import (
    DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
    EmbeddingSettings,
)
from rag_anchor.infra.llm import (
    HTTPTimeoutError,
    HTTPTransport,
    HTTPTransportError,
    UrllibHTTPTransport,
)


class EmbeddingError(RuntimeError):
    """Base class for embedding-provider failures."""


class EmbeddingDisabledError(EmbeddingError):
    """Semantic retrieval is disabled or not configured (HTTP 503)."""


class EmbeddingServiceError(EmbeddingError):
    """The embedding service could not be reached or returned an error (HTTP 502)."""


class EmbeddingTimeoutError(EmbeddingServiceError):
    """The embedding request exceeded the configured timeout (HTTP 504)."""


class EmbeddingResponseError(EmbeddingError):
    """The embedding service returned an unusable response envelope (HTTP 502)."""


@dataclass(frozen=True, slots=True)
class EmbeddingModelIdentity:
    """Verified identity of the embedding model reported by the server.

    ``name`` is the matching entry reported by ``GET /api/tags`` and ``digest``
    is its content digest. The operator-supplied version label is deliberately
    not part of this value: it is useful for manual invalidation but cannot
    prove which model weights produced a vector.
    """

    name: str
    digest: str


class EmbeddingProvider(Protocol):
    """The capability the semantic index and search services need."""

    @property
    def model_name(self) -> str:
        """Configured model name, recorded in the derived index metadata."""
        ...

    @property
    def model_version(self) -> str:
        """Operator-supplied invalidation label, not proof of model identity."""
        ...

    def resolve_identity(self) -> EmbeddingModelIdentity:
        """Resolve and verify the model digest from the service.

        Called once per explicit index build or semantic query, never per
        batch. Implementations must not cache the result.
        """
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one raw vector per input text, in input order."""
        ...


class DisabledEmbeddingProvider:
    """A provider that fails loudly instead of pretending vectors exist."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def model_name(self) -> str:
        return "disabled"

    @property
    def model_version(self) -> str:
        return "disabled"

    def resolve_identity(self) -> EmbeddingModelIdentity:
        raise EmbeddingDisabledError(self._reason)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingDisabledError(self._reason)


class OllamaEmbeddingProvider:
    """Call Ollama's native ``POST {base_url}/api/embed`` endpoint.

    ``transport`` exists for offline tests via a fake transport. The production
    path uses :class:`rag_anchor.infra.llm.UrllibHTTPTransport`, which disables
    ambient proxies and redirects.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        model_version: str = "",
        api_key: str = "",
        timeout_seconds: float = DEFAULT_EMBEDDING_TIMEOUT_SECONDS,
        transport: HTTPTransport | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._endpoint = f"{base_url.rstrip('/')}/api/embed"
        self._tags_endpoint = f"{base_url.rstrip('/')}/api/tags"
        self._model = model.strip()
        self._model_version = model_version.strip() or "unversioned"
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._transport: HTTPTransport = transport or UrllibHTTPTransport()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def model_version(self) -> str:
        return self._model_version

    def resolve_identity(self) -> EmbeddingModelIdentity:
        """Read the matching model's digest from ``GET {base_url}/api/tags``."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            response = self._transport.get(
                self._tags_endpoint,
                headers=headers,
                timeout=self._timeout_seconds,
            )
        except HTTPTimeoutError as exc:
            raise EmbeddingTimeoutError("Embedding model identity request timed out") from exc
        except HTTPTransportError as exc:
            raise EmbeddingServiceError("Embedding model identity request failed") from exc

        if not 200 <= response.status < 300:
            raise EmbeddingServiceError(
                f"Embedding service returned HTTP {response.status} while listing models"
            )

        try:
            parsed = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingResponseError("Model list response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise EmbeddingResponseError("Model list response was not a JSON object")
        models = parsed.get("models")
        if not isinstance(models, list):
            raise EmbeddingResponseError("Model list response did not contain models")

        wanted = _wanted_model_names(self._model)
        matches = [
            entry
            for entry in models
            if isinstance(entry, Mapping) and _reported_model_names(entry) & wanted
        ]
        if not matches:
            raise EmbeddingResponseError(
                f"Embedding model {self._model!r} is not reported by the service"
            )

        names: set[str] = set()
        digests: set[str] = set()
        for entry in matches:
            name = _reported_model_name(entry)
            digest = entry.get("digest")
            if name is None:
                raise EmbeddingResponseError("Embedding model identity has no name")
            if not isinstance(digest, str) or not digest.strip():
                raise EmbeddingResponseError("Embedding model identity has no digest")
            names.add(name)
            digests.add(digest.strip())

        if len(digests) != 1:
            raise EmbeddingResponseError("Embedding model identity is ambiguous")
        return EmbeddingModelIdentity(name=next(iter(names)), digest=next(iter(digests)))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        for item in items:
            if not isinstance(item, str):
                raise TypeError("embedding input must be a string")
            if not item.strip():
                raise ValueError("embedding input must not be empty or whitespace-only")

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model, "input": items}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        try:
            response = self._transport.post(
                self._endpoint,
                headers=headers,
                body=body,
                timeout=self._timeout_seconds,
            )
        except HTTPTimeoutError as exc:
            raise EmbeddingTimeoutError("Embedding request timed out") from exc
        except HTTPTransportError as exc:
            raise EmbeddingServiceError("Embedding service request failed") from exc

        if not 200 <= response.status < 300:
            raise EmbeddingServiceError(f"Embedding service returned HTTP {response.status}")

        try:
            parsed = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EmbeddingResponseError("Embedding response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise EmbeddingResponseError("Embedding response was not a JSON object")
        embeddings = parsed.get("embeddings")
        if not isinstance(embeddings, list):
            raise EmbeddingResponseError("Embedding response did not contain embeddings")
        if len(embeddings) != len(items):
            raise EmbeddingResponseError(
                "Embedding response did not return exactly one vector per input"
            )

        vectors: list[list[float]] = []
        dimension: int | None = None
        for embedding in embeddings:
            vector = _validate_vector(embedding)
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise EmbeddingResponseError(
                    "Embedding response contained inconsistent vector dimensions"
                )
            vectors.append(vector)
        return vectors


def _validate_vector(value: object) -> list[float]:
    if not isinstance(value, list):
        raise EmbeddingResponseError("Embedding response contained a non-list vector")
    if not value:
        raise EmbeddingResponseError("Embedding response contained an empty vector")

    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise EmbeddingResponseError(
                "Embedding response contained a non-numeric vector component"
            )
        number = float(component)
        if not math.isfinite(number):
            raise EmbeddingResponseError("Embedding response contained NaN or infinite values")
        vector.append(number)
    return vector


def _wanted_model_names(model: str) -> set[str]:
    """Return tag spellings that identify the configured model.

    Ollama usually reports an untagged pull as ``name:latest``; accepting both
    spellings keeps identity resolution independent of that presentation.
    """
    wanted = {model}
    if ":" in model:
        base, _, tag = model.rpartition(":")
        if tag == "latest" and base:
            wanted.add(base)
    else:
        wanted.add(f"{model}:latest")
    return wanted


def _reported_model_names(entry: Mapping[str, object]) -> set[str]:
    names: set[str] = set()
    for key in ("name", "model"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            names.add(value.strip())
    return names


def _reported_model_name(entry: Mapping[str, object]) -> str | None:
    for key in ("name", "model"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_embedding_provider(settings: EmbeddingSettings) -> EmbeddingProvider:
    """Build the configured provider without probing the network."""
    if not settings.enabled:
        return DisabledEmbeddingProvider(
            "Semantic retrieval is disabled; set RAG_ANCHOR_EMBEDDING_ENABLED=true "
            "after the embedding model is verified"
        )
    if not settings.configured:
        return DisabledEmbeddingProvider(
            "Semantic retrieval requires RAG_ANCHOR_EMBEDDING_BASE_URL "
            "and RAG_ANCHOR_EMBEDDING_MODEL"
        )
    return OllamaEmbeddingProvider(
        base_url=settings.base_url,
        model=settings.model,
        model_version=settings.resolved_model_version,
        api_key=settings.api_key,
        timeout_seconds=settings.timeout_seconds,
    )


__all__ = [
    "DisabledEmbeddingProvider",
    "EmbeddingDisabledError",
    "EmbeddingError",
    "EmbeddingModelIdentity",
    "EmbeddingProvider",
    "EmbeddingResponseError",
    "EmbeddingServiceError",
    "EmbeddingTimeoutError",
    "OllamaEmbeddingProvider",
    "build_embedding_provider",
]
