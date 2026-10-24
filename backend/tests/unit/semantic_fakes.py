"""Deterministic embedding doubles for offline semantic tests.

These are engineering fixtures only. The keyword vectors below are designed to
exercise the plumbing (ranking, provenance, invalidation); they say nothing
about real embedding quality and are never presented as a benchmark result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_anchor.infra.embedding import EmbeddingModelIdentity

VECTOR_DIMENSION = 8
_UNKNOWN_INDEX = 6
_MISC_INDEX = 7

_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("stage", ("stage", "stages", "阶段", "步骤")),
    ("retrieval", ("retrieval", "retrieve", "检索", "搜索")),
    ("evidence", ("evidence", "citation", "证据", "引用")),
    ("photo", ("photo", "image", "图像", "照片")),
    ("security", ("security", "secure", "安全")),
    ("latency", ("latency", "延迟")),
)


def keyword_vector(text: str) -> list[float]:
    """Return a small deterministic cross-lingual keyword vector."""
    lowered = text.lower()
    vector = [0.0] * VECTOR_DIMENSION
    matched = False
    for index, (_, words) in enumerate(_KEYWORDS):
        if any(word in lowered for word in words):
            vector[index] = 1.0
            matched = True
    if not matched:
        vector[_UNKNOWN_INDEX] = 1.0
    vector[_MISC_INDEX] = 0.1
    return vector


def fake_identity(model_name: str, model_version: str) -> EmbeddingModelIdentity:
    """Deterministic identity double; it never represents a real model."""
    return EmbeddingModelIdentity(
        name=model_name,
        digest=f"fake-digest:{model_version}",
    )


@dataclass
class FakeEmbeddingProvider:
    """Keyword provider double that records every embedded batch."""

    model_name: str = "fake-bge-m3"
    model_version: str = "v1"
    calls: list[list[str]] = field(default_factory=list)

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [keyword_vector(text) for text in texts]


@dataclass
class VectorMapEmbeddingProvider:
    """Provider double returning explicitly scripted vectors by input text."""

    vectors: dict[str, list[float]]
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"
    calls: list[list[str]] = field(default_factory=list)

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]
