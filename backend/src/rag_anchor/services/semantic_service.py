"""Derived semantic index and brute-force cosine search.

The semantic index is *derived* and may be deleted and rebuilt at any time. It
never becomes the source of truth for text or offsets:

* candidates are built from persisted chunks, verified against stored page
  text, and stored with exact Python string offsets;
* vectors are L2-normalized before persistence, so cosine similarity is a dot
  product;
* the metadata records the model name/version, dimension, normalization rule
  and the source text-index fingerprint;
* a rebuild of the text index, a document deletion or a provider change
  invalidates the vectors, and a stale index is rejected instead of used.

An embedding only ranks candidates. It cannot create, move or widen a PDF span;
a candidate that is not an exact slice of its stored page text is never
returned.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from rag_anchor.config import EmbeddingSettings
from rag_anchor.domain.text import TEXT_EXTRACTION_VERSION
from rag_anchor.infra.database import (
    DocumentIndexRecord,
    DocumentRepository,
    SemanticCandidateRecord,
    SemanticIndexRecord,
)
from rag_anchor.infra.embedding import (
    DisabledEmbeddingProvider,
    EmbeddingDisabledError,
    EmbeddingProvider,
    EmbeddingResponseError,
)
from rag_anchor.services.document_service import (
    DocumentNotFoundError,
    DocumentService,
)
from rag_anchor.services.retrieval_service import MAX_QUERY_LENGTH
from rag_anchor.services.semantic_candidates import (
    SEMANTIC_CANDIDATE_BUILDER_VERSION,
    SemanticCandidate,
    build_semantic_candidates,
)

SEMANTIC_INDEX_FORMAT_VERSION = 1
"""Version of the persisted vector-index layout and ranking semantics."""

NORMALIZATION_L2 = "l2"
"""The only vector normalization rule stored in this version."""

DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MIN_VECTOR_NORM = 1e-12
L2_NORM_TOLERANCE = 1e-3
"""Allowed deviation from unit norm for stored float32 vectors.

Vectors are normalized in float64 and stored as float32, so valid rows are
unit norm to roughly 1e-7. The wider tolerance avoids rejecting harmless
float32 rounding while still rejecting vectors that were never normalized.
"""


class SemanticIndexError(RuntimeError):
    """Base class for derived semantic index failures."""


class SemanticBaseIndexUnavailableError(SemanticIndexError):
    """The document has no usable text/chunk index (HTTP 409)."""


class SemanticIndexUnavailableError(SemanticIndexError):
    """The document has no derived semantic index (HTTP 409)."""


class SemanticIndexStaleError(SemanticIndexError):
    """The derived index no longer matches its source or provider (HTTP 409)."""


class SemanticIndexEmptyError(SemanticIndexError):
    """The document has no indexable text candidates (HTTP 409)."""


class SemanticIndexPersistenceError(SemanticIndexError):
    """The derived index transaction could not be written (HTTP 500)."""


@dataclass(frozen=True, slots=True)
class SemanticIndexStats:
    """Statistics of one completed semantic index build."""

    document_id: UUID
    model_name: str
    model_version: str
    model_digest: str
    index_version: int
    candidate_version: int
    dimension: int
    normalization: str
    source_fingerprint: str
    candidate_count: int
    built_at: datetime


@dataclass(frozen=True, slots=True)
class SemanticMatch:
    """One ranked candidate with an exact page-text span."""

    document_id: UUID
    chunk_id: str
    page_index: int
    page_number: int
    chunk_index: int
    start_offset: int
    end_offset: int
    source_text: str
    score: float
    extraction_version: str


@dataclass(frozen=True, slots=True)
class SemanticSearchResult:
    """Ranked matches plus the unpaginated number above the score threshold."""

    query: str
    min_score: float
    total: int
    matches: list[SemanticMatch]


class SemanticIndexService:
    """Build, validate and query one document's derived vector index."""

    def __init__(
        self,
        document_service: DocumentService,
        embedding_provider: EmbeddingProvider,
        *,
        settings: EmbeddingSettings | None = None,
    ) -> None:
        self._document_service = document_service
        self._repository: DocumentRepository = document_service.repository
        self._embedding_provider = embedding_provider
        self._settings = settings or EmbeddingSettings()

    @property
    def repository(self) -> DocumentRepository:
        return self._repository

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        return self._embedding_provider

    def build_index(self, document_id: UUID) -> SemanticIndexStats:
        """Embed every candidate of one document and replace its vector index."""
        # A missing document is a 404 even when semantic retrieval is disabled,
        # so an unknown ID is never reported as a configuration problem.
        if self._document_service.get_document(document_id) is None:
            raise DocumentNotFoundError("Document not found")
        if isinstance(self._embedding_provider, DisabledEmbeddingProvider):
            raise EmbeddingDisabledError("Semantic retrieval is disabled")

        base = self._repository.get_document_index_metadata(document_id)
        if base is None:
            raise SemanticBaseIndexUnavailableError(
                "Document has not been indexed; build the text index first"
            )
        if base.extraction_version != TEXT_EXTRACTION_VERSION:
            raise SemanticIndexStaleError(
                "Stored index was built with a different extraction version"
            )

        pages = self._repository.get_document_page_texts(document_id)
        chunks = self._repository.get_document_chunks(document_id)
        if not pages or not chunks:
            raise SemanticIndexEmptyError("Document has no indexable text")

        candidates = build_semantic_candidates(pages, chunks)
        if not candidates:
            raise SemanticIndexEmptyError("Document has no indexable text")

        # Resolve identity once, immediately before embedding. The digest is
        # persisted with the vectors, so a later query can prove that the same
        # model weights produced them.
        identity = self._embedding_provider.resolve_identity()
        vectors = self._embed_candidates(candidates)

        built_at = datetime.now(UTC)
        dimension = len(vectors[0])
        records = [
            SemanticCandidateRecord(
                candidate_id=candidate.candidate_id,
                document_id=candidate.document_id,
                chunk_id=candidate.chunk_id,
                page_index=candidate.page_index,
                page_number=candidate.page_number,
                chunk_index=candidate.chunk_index,
                start_offset=candidate.start_offset,
                end_offset=candidate.end_offset,
                source_text=candidate.text,
                extraction_version=candidate.extraction_version,
                vector=_encode_vector(vector),
            )
            for candidate, vector in zip(candidates, vectors, strict=True)
        ]
        metadata = SemanticIndexRecord(
            document_id=document_id,
            model_name=self._embedding_provider.model_name,
            model_version=self._embedding_provider.model_version,
            model_digest=identity.digest,
            dimension=dimension,
            normalization=NORMALIZATION_L2,
            index_version=SEMANTIC_INDEX_FORMAT_VERSION,
            candidate_version=SEMANTIC_CANDIDATE_BUILDER_VERSION,
            source_fingerprint=base.fingerprint,
            source_extraction_version=base.extraction_version,
            candidate_count=len(records),
            built_at=built_at,
        )

        try:
            self._repository.replace_semantic_index(
                document_id,
                metadata=metadata,
                candidates=records,
            )
        except KeyError as exc:
            # The document was deleted after the build started. Report the same
            # 404 as an unknown ID instead of leaking a KeyError as a 500.
            raise DocumentNotFoundError("Document not found") from exc
        except sqlite3.Error as exc:
            raise SemanticIndexPersistenceError("Failed to persist the semantic index") from exc

        return SemanticIndexStats(
            document_id=document_id,
            model_name=metadata.model_name,
            model_version=metadata.model_version,
            model_digest=metadata.model_digest,
            index_version=metadata.index_version,
            candidate_version=metadata.candidate_version,
            dimension=metadata.dimension,
            normalization=metadata.normalization,
            source_fingerprint=metadata.source_fingerprint,
            candidate_count=metadata.candidate_count,
            built_at=built_at,
        )

    def search(
        self,
        document_id: UUID,
        query: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
    ) -> SemanticSearchResult:
        """Return the highest-scoring valid candidates above ``min_score``."""
        _validate_query(query)
        _validate_top_k(top_k)
        threshold = self._settings.min_score if min_score is None else min_score
        _validate_min_score(threshold)

        # A missing document is a 404 even when semantic retrieval is disabled.
        if self._document_service.get_document(document_id) is None:
            raise DocumentNotFoundError("Document not found")
        if isinstance(self._embedding_provider, DisabledEmbeddingProvider):
            raise EmbeddingDisabledError("Semantic retrieval is disabled")

        base = self._repository.get_document_index_metadata(document_id)
        if base is None:
            raise SemanticBaseIndexUnavailableError(
                "Document has not been indexed; build the text index first"
            )
        if base.extraction_version != TEXT_EXTRACTION_VERSION:
            raise SemanticIndexStaleError(
                "Stored index was built with a different extraction version"
            )

        semantic = self._repository.get_semantic_index_metadata(document_id)
        if semantic is None:
            raise SemanticIndexUnavailableError("Semantic index has not been built")
        self._require_compatible(semantic, base)

        candidates = self._repository.list_semantic_candidates(document_id)
        if len(candidates) != semantic.candidate_count:
            raise SemanticIndexStaleError(
                "Semantic index candidate count does not match its metadata"
            )
        self._validate_candidates(document_id, candidates)

        # Local checks passed; prove that the active server-side model still
        # has the exact digest recorded when the vectors were built.
        identity = self._embedding_provider.resolve_identity()
        if semantic.model_digest != identity.digest:
            raise SemanticIndexStaleError(
                "Embedding model identity changed; rebuild the semantic index"
            )

        query_vector = self._embed_query(query)
        if len(query_vector) != semantic.dimension:
            raise SemanticIndexStaleError(
                "Semantic index was built with a different vector dimension"
            )

        scored: list[tuple[float, SemanticCandidateRecord]] = []
        for candidate in candidates:
            vector = _decode_vector(candidate.vector, semantic.dimension)
            score = _cosine(query_vector, vector)
            if score >= threshold:
                scored.append((score, candidate))

        scored.sort(key=_ranking_key)
        matches = [
            SemanticMatch(
                document_id=document_id,
                chunk_id=candidate.chunk_id,
                page_index=candidate.page_index,
                page_number=candidate.page_number,
                chunk_index=candidate.chunk_index,
                start_offset=candidate.start_offset,
                end_offset=candidate.end_offset,
                source_text=candidate.source_text,
                score=score,
                extraction_version=candidate.extraction_version,
            )
            for score, candidate in scored[:top_k]
        ]
        return SemanticSearchResult(
            query=query,
            min_score=threshold,
            total=len(scored),
            matches=matches,
        )

    def _embed_candidates(
        self,
        candidates: Sequence[SemanticCandidate],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        dimension: int | None = None
        batch_size = self._settings.batch_size
        items = list(candidates)
        for batch in _batches(items, batch_size):
            texts = [candidate.text for candidate in batch]
            raw_vectors = self._embedding_provider.embed(texts)
            if len(raw_vectors) != len(batch):
                raise EmbeddingResponseError(
                    "Embedding provider did not return exactly one vector per candidate"
                )
            for raw_vector in raw_vectors:
                vector = _normalize_vector(raw_vector)
                if dimension is None:
                    dimension = len(vector)
                elif len(vector) != dimension:
                    raise EmbeddingResponseError(
                        "Embedding provider returned inconsistent vector dimensions"
                    )
                vectors.append(vector)
        return vectors

    def _embed_query(self, query: str) -> list[float]:
        vectors = self._embedding_provider.embed([query])
        if len(vectors) != 1:
            raise EmbeddingResponseError(
                "Embedding provider did not return exactly one query vector"
            )
        return _normalize_vector(vectors[0])

    def _require_compatible(
        self,
        semantic: SemanticIndexRecord,
        base: DocumentIndexRecord,
    ) -> None:
        if semantic.index_version != SEMANTIC_INDEX_FORMAT_VERSION:
            raise SemanticIndexStaleError(
                "Semantic index uses a different index format; rebuild required"
            )
        if semantic.candidate_version != SEMANTIC_CANDIDATE_BUILDER_VERSION:
            raise SemanticIndexStaleError(
                "Semantic index was built with a different candidate builder; rebuild required"
            )
        if not semantic.model_digest:
            raise SemanticIndexStaleError(
                "Semantic index has no verified embedding model identity; rebuild required"
            )
        if (
            semantic.model_name != self._embedding_provider.model_name
            or semantic.model_version != self._embedding_provider.model_version
        ):
            raise SemanticIndexStaleError(
                "Semantic index was built with a different embedding model"
            )
        if semantic.normalization != NORMALIZATION_L2:
            raise SemanticIndexStaleError(
                "Semantic index uses an unsupported vector normalization rule"
            )
        if semantic.source_fingerprint != base.fingerprint:
            raise SemanticIndexStaleError("Semantic index was built from a different text index")
        if semantic.source_extraction_version != TEXT_EXTRACTION_VERSION:
            raise SemanticIndexStaleError(
                "Semantic index was built with a different extraction version"
            )

    def _validate_candidates(
        self,
        document_id: UUID,
        candidates: Sequence[SemanticCandidateRecord],
    ) -> None:
        pages = {
            page.page_index: page for page in self._repository.get_document_page_texts(document_id)
        }
        chunks = {
            chunk.chunk_id: chunk for chunk in self._repository.get_document_chunks(document_id)
        }
        if not pages or not chunks:
            raise SemanticIndexStaleError("Text index is missing while a semantic index exists")

        for candidate in candidates:
            if candidate.document_id != document_id:
                raise SemanticIndexStaleError(
                    "Semantic index contains a candidate from another document"
                )
            page = pages.get(candidate.page_index)
            chunk = chunks.get(candidate.chunk_id)
            if page is None or chunk is None:
                raise SemanticIndexStaleError(
                    "Semantic index references text rows that no longer exist"
                )
            if (
                chunk.document_id != str(document_id)
                or chunk.page_index != candidate.page_index
                or chunk.page_number != candidate.page_number
                or chunk.chunk_index != candidate.chunk_index
                or chunk.extraction_version != TEXT_EXTRACTION_VERSION
                or candidate.extraction_version != TEXT_EXTRACTION_VERSION
            ):
                raise SemanticIndexStaleError("Semantic index references an incompatible chunk")
            if candidate.page_number != page.page_number:
                raise SemanticIndexStaleError(
                    "Semantic candidate page number does not match the stored page"
                )
            if page.text[chunk.start_offset : chunk.end_offset] != chunk.text:
                raise SemanticIndexStaleError("Stored chunk no longer matches its page text")
            if (
                candidate.start_offset < chunk.start_offset
                or candidate.end_offset > chunk.end_offset
            ):
                raise SemanticIndexStaleError("Semantic candidate is outside its source chunk")
            if page.text[candidate.start_offset : candidate.end_offset] != candidate.source_text:
                raise SemanticIndexStaleError(
                    "Semantic candidate is not an exact slice of the stored page text"
                )


def _batches(
    items: Sequence[SemanticCandidate],
    size: int,
) -> Iterable[Sequence[SemanticCandidate]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalize_vector(value: Sequence[float]) -> list[float]:
    components: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise EmbeddingResponseError("Embedding vector contained a non-numeric value")
        number = float(component)
        if not math.isfinite(number):
            raise EmbeddingResponseError("Embedding vector contained NaN or infinity")
        components.append(number)
    if not components:
        raise EmbeddingResponseError("Embedding vector was empty")

    norm = math.sqrt(math.fsum(component * component for component in components))
    if not math.isfinite(norm) or norm < MIN_VECTOR_NORM:
        raise EmbeddingResponseError("Embedding vector has a zero or invalid norm")
    return [component / norm for component in components]


def _encode_vector(vector: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _decode_vector(blob: bytes, expected_dimension: int) -> list[float]:
    """Decode one persisted vector and reject anything unusable.

    A malformed, wrongly sized, non-finite, zero-norm or clearly
    non-L2-normalized vector makes the whole index stale. Invalid values never
    reach cosine ranking, so a query cannot return a partial or NaN score.
    """
    if not blob or len(blob) % 4 != 0:
        raise SemanticIndexStaleError("Semantic index contains a malformed vector")
    dimension = len(blob) // 4
    if dimension != expected_dimension:
        raise SemanticIndexStaleError(
            "Semantic index contains a vector with an unexpected dimension"
        )

    vector = list(struct.unpack(f"<{dimension}f", blob))
    if not all(math.isfinite(component) for component in vector):
        raise SemanticIndexStaleError(
            "Semantic index contains a vector with NaN or infinite values"
        )

    norm = math.sqrt(math.fsum(component * component for component in vector))
    if not math.isfinite(norm) or norm < MIN_VECTOR_NORM:
        raise SemanticIndexStaleError("Semantic index contains a zero-norm or invalid vector")
    if abs(norm - 1.0) > L2_NORM_TOLERANCE:
        raise SemanticIndexStaleError("Semantic index contains a non-normalized vector")
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise SemanticIndexStaleError("Semantic index vector dimension mismatch")
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _ranking_key(pair: tuple[float, SemanticCandidateRecord]) -> tuple[float, int, int, int, str]:
    score, candidate = pair
    return (
        -score,
        candidate.page_index,
        candidate.chunk_index,
        candidate.start_offset,
        candidate.candidate_id,
    )


def _validate_query(query: object) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")


def _validate_top_k(top_k: object) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")


def _validate_min_score(min_score: object) -> None:
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        raise TypeError("min_score must be a number")
    if not math.isfinite(float(min_score)) or not -1.0 <= float(min_score) <= 1.0:
        raise ValueError("min_score must be between -1 and 1")


__all__ = [
    "DEFAULT_TOP_K",
    "L2_NORM_TOLERANCE",
    "MAX_TOP_K",
    "NORMALIZATION_L2",
    "SEMANTIC_INDEX_FORMAT_VERSION",
    "SemanticBaseIndexUnavailableError",
    "SemanticIndexEmptyError",
    "SemanticIndexError",
    "SemanticIndexPersistenceError",
    "SemanticIndexService",
    "SemanticIndexStaleError",
    "SemanticIndexStats",
    "SemanticIndexUnavailableError",
    "SemanticMatch",
    "SemanticSearchResult",
]
