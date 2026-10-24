"""Derived semantic index tests against real SQLite rows.

The embedding provider is a deterministic keyword double, so these tests check
plumbing and provenance, not embedding quality. Every assertion about geometry
is made against the stored page text, never against provider output.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from semantic_fakes import (
    FakeEmbeddingProvider,
    VectorMapEmbeddingProvider,
    fake_identity,
    keyword_vector,
)

from rag_anchor.config import EmbeddingSettings, Settings
from rag_anchor.domain import PageText, TextChunk, build_chunk_id
from rag_anchor.infra import DocumentIndexRecord, DocumentRecord
from rag_anchor.infra.database import DocumentRepository
from rag_anchor.infra.embedding import (
    DisabledEmbeddingProvider,
    EmbeddingDisabledError,
    EmbeddingModelIdentity,
    EmbeddingResponseError,
    EmbeddingServiceError,
)
from rag_anchor.services import DocumentNotFoundError, DocumentService
from rag_anchor.services.semantic_candidates import SEMANTIC_CANDIDATE_BUILDER_VERSION
from rag_anchor.services.semantic_service import (
    SEMANTIC_INDEX_FORMAT_VERSION,
    SemanticBaseIndexUnavailableError,
    SemanticIndexEmptyError,
    SemanticIndexService,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
)

STAGES_TEXT = "The project has three stages."
PHOTO_TEXT = "The photo is stored separately."
CHINESE_QUESTION = "这个项目有哪些阶段？"  # noqa: RUF001


def _prepare(tmp_path: Path) -> tuple[DocumentService, Path]:
    settings = Settings(data_dir=tmp_path / "data")
    service = DocumentService(settings)
    service.initialize()
    return service, settings.database_path


def _seed_document(
    service: DocumentService,
    pages: dict[int, str],
    *,
    fingerprint: str = "fingerprint",
) -> UUID:
    """Insert one document with one full-page chunk per page."""
    document_id = uuid4()
    document_key = str(document_id)
    page_models = [
        PageText(
            document_id=document_key,
            page_index=page_index,
            page_number=page_index + 1,
            text=text,
        )
        for page_index, text in sorted(pages.items())
    ]
    chunk_models = [
        TextChunk(
            chunk_id=build_chunk_id(
                document_id=document_key,
                page_index=page.page_index,
                start_offset=0,
                end_offset=len(page.text),
            ),
            document_id=document_key,
            page_index=page.page_index,
            page_number=page.page_number,
            start_offset=0,
            end_offset=len(page.text),
            text=page.text,
            chunk_index=0,
        )
        for page in page_models
    ]
    service.repository.create(
        DocumentRecord(
            document_id=document_id,
            filename="seed.pdf",
            size_bytes=1,
            page_count=len(pages),
            created_at=datetime.now(UTC),
        )
    )
    service.repository.replace_document_index(
        document_id,
        pages=page_models,
        chunks=chunk_models,
        metadata=DocumentIndexRecord(
            document_id=document_id,
            pdf_sha256="0" * 64,
            extraction_version="pdfium-text-v1",
            pdfium_version="test",
            max_chars=800,
            overlap_chars=0,
            fingerprint=fingerprint,
            page_count=len(pages),
            chunk_count=len(chunk_models),
            filtered_chunk_count=0,
            indexed_at=datetime.now(UTC),
        ),
    )
    return document_id


def _build(
    service: DocumentService,
    provider,
    document_id: UUID,
    *,
    settings: EmbeddingSettings | None = None,
):
    semantic = SemanticIndexService(service, provider, settings=settings)
    return semantic, semantic.build_index(document_id)


def _page_text(service: DocumentService, document_id: UUID, page_index: int) -> str:
    pages = {
        page.page_index: page.text
        for page in service.repository.get_document_page_texts(document_id)
    }
    return pages[page_index]


def _overwrite_vector(database_path: Path, document_id: UUID, blob: bytes) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_candidates SET vector = ? WHERE document_id = ?",
            (blob, str(document_id)),
        )


def _encoded(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def test_build_persists_metadata_and_normalized_vectors(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    provider = FakeEmbeddingProvider()

    semantic, stats = _build(service, provider, document_id)

    assert stats.document_id == document_id
    assert stats.model_name == "fake-bge-m3"
    assert stats.model_version == "v1"
    assert stats.model_digest == "fake-digest:v1"
    assert stats.index_version == SEMANTIC_INDEX_FORMAT_VERSION
    assert stats.candidate_version == SEMANTIC_CANDIDATE_BUILDER_VERSION
    assert stats.dimension == 8
    assert stats.normalization == "l2"
    assert stats.source_fingerprint == "fingerprint"
    assert stats.candidate_count == 1

    metadata = semantic.repository.get_semantic_index_metadata(document_id)
    assert metadata is not None
    assert metadata.model_name == "fake-bge-m3"
    assert metadata.model_version == "v1"
    assert metadata.model_digest == "fake-digest:v1"
    assert metadata.index_version == SEMANTIC_INDEX_FORMAT_VERSION
    assert metadata.candidate_version == SEMANTIC_CANDIDATE_BUILDER_VERSION
    assert metadata.dimension == 8
    assert metadata.normalization == "l2"
    assert metadata.source_fingerprint == "fingerprint"
    assert metadata.candidate_count == 1

    candidates = semantic.repository.list_semantic_candidates(document_id)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source_text == STAGES_TEXT
    assert candidate.start_offset == 0
    assert candidate.end_offset == len(STAGES_TEXT)
    assert candidate.extraction_version == "pdfium-text-v1"

    with sqlite3.connect(database_path) as connection:
        raw = connection.execute(
            "SELECT vector FROM semantic_candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()[0]
    vector = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert len(vector) == 8
    assert math.isclose(
        math.sqrt(math.fsum(value * value for value in vector)),
        1.0,
        rel_tol=1e-6,
    )


def test_cross_lingual_query_ranks_english_candidate_with_exact_offsets(
    tmp_path: Path,
) -> None:
    service, _ = _prepare(tmp_path)
    page_text = f"{STAGES_TEXT} {PHOTO_TEXT}"
    document_id = _seed_document(service, {0: page_text})
    provider = FakeEmbeddingProvider()
    semantic, _ = _build(service, provider, document_id)

    result = semantic.search(document_id, CHINESE_QUESTION, top_k=2)

    assert result.total >= 1
    assert result.matches
    best = result.matches[0]
    assert best.source_text == STAGES_TEXT
    assert best.page_index == 0
    assert best.page_number == 1
    assert best.score > 0.3
    assert page_text[best.start_offset : best.end_offset] == best.source_text

    chunks = service.repository.get_document_chunks(document_id)
    chunk = next(item for item in chunks if item.chunk_id == best.chunk_id)
    assert chunk.start_offset <= best.start_offset
    assert best.end_offset <= chunk.end_offset


def test_search_embeds_only_the_query_and_ranks_real_sentences(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    provider = FakeEmbeddingProvider()
    semantic, _ = _build(service, provider, document_id)
    assert provider.calls == [[STAGES_TEXT]]

    semantic.search(document_id, "阶段", top_k=5)

    assert provider.calls == [[STAGES_TEXT], ["阶段"]]


def test_equal_scores_use_stable_source_order(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    page_text = "Stage one. Stage two."
    document_id = _seed_document(service, {0: page_text})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    first = semantic.search(document_id, "阶段", top_k=2)
    second = semantic.search(document_id, "阶段", top_k=2)

    assert [match.source_text for match in first.matches] == ["Stage one.", "Stage two."]
    assert first.matches[0].score == first.matches[1].score
    assert [match.start_offset for match in first.matches] == [0, 11]
    assert [(match.chunk_id, match.start_offset) for match in second.matches] == [
        (match.chunk_id, match.start_offset) for match in first.matches
    ]


def test_top_k_limits_returned_matches_but_total_reports_all(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: "Stage one. Stage two. Stage three."})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    result = semantic.search(document_id, "阶段", top_k=1)

    assert result.total == 3
    assert len(result.matches) == 1


def test_unrelated_query_below_threshold_returns_nothing(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    result = semantic.search(document_id, "quantum chromodynamics", top_k=5)

    assert result.total == 0
    assert result.matches == []


def test_min_score_override_can_admit_more_or_fewer_matches(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    page_text = "The project has three stages: ingestion, retrieval, and evidence verification."
    document_id = _seed_document(service, {0: page_text})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    strict = semantic.search(document_id, CHINESE_QUESTION, top_k=5, min_score=0.9)
    loose = semantic.search(document_id, CHINESE_QUESTION, top_k=5, min_score=0.5)

    assert strict.matches == []
    assert len(loose.matches) == 1
    assert 0.5 < loose.matches[0].score < 0.9


def test_batching_respects_the_configured_batch_size(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: "Stage one. Stage two. Stage three."})
    provider = FakeEmbeddingProvider()

    _build(
        service,
        provider,
        document_id,
        settings=EmbeddingSettings(batch_size=2),
    )

    assert [len(batch) for batch in provider.calls] == [2, 1]


def test_build_requires_an_existing_document(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    semantic = SemanticIndexService(service, FakeEmbeddingProvider())

    with pytest.raises(DocumentNotFoundError):
        semantic.build_index(uuid4())

    with pytest.raises(DocumentNotFoundError):
        semantic.search(uuid4(), "阶段")


def test_build_requires_the_text_index_first(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = uuid4()
    service.repository.create(
        DocumentRecord(
            document_id=document_id,
            filename="seed.pdf",
            size_bytes=1,
            page_count=1,
            created_at=datetime.now(UTC),
        )
    )
    semantic = SemanticIndexService(service, FakeEmbeddingProvider())

    with pytest.raises(SemanticBaseIndexUnavailableError):
        semantic.build_index(document_id)


def test_document_without_candidates_is_rejected(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: "   "})
    semantic = SemanticIndexService(service, FakeEmbeddingProvider())

    with pytest.raises(SemanticIndexEmptyError):
        semantic.build_index(document_id)


def test_disabled_provider_rejects_build_and_search(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic = SemanticIndexService(
        service, DisabledEmbeddingProvider("Semantic retrieval is disabled")
    )

    with pytest.raises(EmbeddingDisabledError):
        semantic.build_index(document_id)

    with pytest.raises(EmbeddingDisabledError):
        semantic.search(document_id, "阶段")


def test_search_requires_a_built_semantic_index(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic = SemanticIndexService(service, FakeEmbeddingProvider())

    with pytest.raises(SemanticIndexUnavailableError):
        semantic.search(document_id, "阶段")


def test_rebuilding_the_text_index_invalidates_the_semantic_index(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    assert semantic.repository.get_semantic_index_metadata(document_id) is not None

    service.repository.replace_document_index(
        document_id,
        pages=[PageText(str(document_id), 0, 1, "Stage one.")],
        chunks=[
            TextChunk(
                chunk_id=build_chunk_id(
                    document_id=str(document_id),
                    page_index=0,
                    start_offset=0,
                    end_offset=10,
                ),
                document_id=str(document_id),
                page_index=0,
                page_number=1,
                start_offset=0,
                end_offset=10,
                text="Stage one.",
                chunk_index=0,
            )
        ],
        metadata=DocumentIndexRecord(
            document_id=document_id,
            pdf_sha256="0" * 64,
            extraction_version="pdfium-text-v1",
            pdfium_version="test",
            max_chars=800,
            overlap_chars=0,
            fingerprint="rebuilt",
            page_count=1,
            chunk_count=1,
            filtered_chunk_count=0,
            indexed_at=datetime.now(UTC),
        ),
    )

    assert semantic.repository.get_semantic_index_metadata(document_id) is None
    assert semantic.repository.list_semantic_candidates(document_id) == []
    with pytest.raises(SemanticIndexUnavailableError):
        semantic.search(document_id, "阶段")


def test_deleting_the_semantic_index_keeps_the_text_index(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    assert semantic.repository.delete_semantic_index(document_id) is True

    assert semantic.repository.get_document_index_metadata(document_id) is not None
    with pytest.raises(SemanticIndexUnavailableError):
        semantic.search(document_id, "阶段")


def test_provider_version_change_marks_the_index_stale(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    _build(service, FakeEmbeddingProvider(), document_id)

    upgraded = FakeEmbeddingProvider(model_version="v2")
    semantic = SemanticIndexService(service, upgraded)

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_candidate_count_mismatch_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_index_metadata SET candidate_count = 99 WHERE document_id = ?",
            (str(document_id),),
        )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_tampered_candidate_text_is_stale_and_never_returned(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_candidates SET source_text = 'Fabricated span.' WHERE document_id = ?",
            (str(document_id),),
        )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_vector_with_wrong_dimension_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_candidates SET vector = ? WHERE document_id = ?",
            (struct.pack("<2f", 0.5, 0.5), str(document_id)),
        )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


@pytest.mark.parametrize(
    "blob",
    [
        b"",
        b"\x00",
        _encoded([float("nan")] + [0.0] * 7),
        _encoded([float("inf")] + [0.0] * 7),
        _encoded([float("-inf")] + [0.0] * 7),
        _encoded([0.0] * 8),
        _encoded([0.5] * 8),
    ],
    ids=[
        "empty",
        "partial-float",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "zero-norm",
        "unnormalized",
    ],
)
def test_invalid_persisted_vector_is_stale(tmp_path: Path, blob: bytes) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    _overwrite_vector(database_path, document_id, blob)

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_valid_float32_normalized_vector_is_accepted(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    _overwrite_vector(database_path, document_id, _encoded([1 / math.sqrt(8)] * 8))

    result = semantic.search(document_id, "阶段")

    assert result.total == 1
    assert result.matches[0].source_text == STAGES_TEXT


def test_query_dimension_mismatch_is_stale(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    provider = VectorMapEmbeddingProvider(
        vectors={
            STAGES_TEXT: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "阶段": [1.0, 0.0, 0.0, 0.0],
        }
    )
    semantic, _ = _build(service, provider, document_id)

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_source_fingerprint_mismatch_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_index_metadata SET source_fingerprint = 'other' WHERE document_id = ?",
            (str(document_id),),
        )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_unknown_normalization_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_index_metadata SET normalization = 'none' WHERE document_id = ?",
            (str(document_id),),
        )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def _tamper_metadata(database_path: Path, document_id: UUID, assignment: str) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"UPDATE semantic_index_metadata SET {assignment} WHERE document_id = ?",
            (str(document_id),),
        )


def test_index_format_version_mismatch_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    _tamper_metadata(database_path, document_id, "index_version = 99")

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_candidate_builder_version_mismatch_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    _tamper_metadata(database_path, document_id, "candidate_version = 99")

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_missing_model_identity_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    _tamper_metadata(database_path, document_id, "model_digest = ''")

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_metadata_without_versions_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    _tamper_metadata(
        database_path,
        document_id,
        "index_version = 0, candidate_version = 0, model_digest = ''",
    )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_initialize_migrates_an_old_semantic_metadata_table(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database_path = data_dir / "metadata.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE semantic_index_metadata (
                document_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                normalization TEXT NOT NULL,
                source_fingerprint TEXT NOT NULL,
                source_extraction_version TEXT NOT NULL,
                candidate_count INTEGER NOT NULL,
                built_at TEXT NOT NULL
            );
            INSERT INTO semantic_index_metadata VALUES
                ('doc', 'old-model', 'v1', 8, 'l2', 'fp', 'ev', 0,
                 '2020-01-01T00:00:00+00:00');
            """
        )

    service = DocumentService(Settings(data_dir=data_dir))
    service.initialize()

    with sqlite3.connect(database_path) as connection:
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(semantic_index_metadata)")
        }
        row = connection.execute(
            "SELECT model_digest, index_version, candidate_version "
            "FROM semantic_index_metadata WHERE document_id = 'doc'"
        ).fetchone()
    assert {"model_digest", "index_version", "candidate_version"} <= columns
    assert row == ("", 0, 0)


def test_candidate_chunk_index_mismatch_is_stale(tmp_path: Path) -> None:
    service, database_path = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE semantic_candidates SET chunk_index = 99 WHERE document_id = ?",
            (str(document_id),),
        )

    with pytest.raises(SemanticIndexStaleError):
        semantic.search(document_id, "阶段")


def test_candidate_page_number_mismatch_is_stale(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)
    candidate = semantic.repository.list_semantic_candidates(document_id)[0]
    tampered = replace(candidate, page_number=candidate.page_number + 1)

    with pytest.raises(SemanticIndexStaleError):
        semantic._validate_candidates(document_id, [tampered])


def test_document_deleted_during_build_is_reported_as_not_found(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic = SemanticIndexService(
        service,
        _DeletingProvider(service.repository, document_id),
    )

    with pytest.raises(DocumentNotFoundError):
        semantic.build_index(document_id)

    assert service.repository.get(document_id) is None
    assert service.repository.get_semantic_index_metadata(document_id) is None


def test_inconsistent_embedding_dimensions_abort_the_build(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    page_text = f"{STAGES_TEXT} {PHOTO_TEXT}"
    document_id = _seed_document(service, {0: page_text})
    provider = VectorMapEmbeddingProvider(
        vectors={
            STAGES_TEXT: [1.0] * 8,
            PHOTO_TEXT: [1.0] * 4,
        }
    )
    semantic = SemanticIndexService(service, provider)

    with pytest.raises(EmbeddingResponseError):
        semantic.build_index(document_id)

    assert semantic.repository.get_semantic_index_metadata(document_id) is None


def test_zero_norm_vector_aborts_the_build(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    provider = VectorMapEmbeddingProvider(vectors={STAGES_TEXT: [0.0] * 8})
    semantic = SemanticIndexService(service, provider)

    with pytest.raises(EmbeddingResponseError):
        semantic.build_index(document_id)

    assert semantic.repository.get_semantic_index_metadata(document_id) is None


@dataclass
class _ToggleProvider:
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"
    failing: bool = False

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts):
        if self.failing:
            raise EmbeddingServiceError("embedding service unavailable")
        return [keyword_vector(text) for text in texts]


@dataclass
class _DeletingProvider:
    """Deletes the document while embedding to simulate a build/delete race."""

    repository: DocumentRepository
    document_id: UUID
    model_name: str = "fake-bge-m3"
    model_version: str = "v1"

    def resolve_identity(self) -> EmbeddingModelIdentity:
        return fake_identity(self.model_name, self.model_version)

    def embed(self, texts):
        self.repository.delete(self.document_id)
        return [keyword_vector(text) for text in texts]


def test_failed_rebuild_keeps_the_previous_index_usable(tmp_path: Path) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    provider = _ToggleProvider()
    semantic, _ = _build(service, provider, document_id)
    pages_before = service.repository.get_document_page_texts(document_id)
    chunks_before = service.repository.get_document_chunks(document_id)
    metadata_before = semantic.repository.get_semantic_index_metadata(document_id)

    provider.failing = True
    with pytest.raises(EmbeddingServiceError):
        semantic.build_index(document_id)

    assert service.repository.get_document_page_texts(document_id) == pages_before
    assert service.repository.get_document_chunks(document_id) == chunks_before
    assert semantic.repository.get_semantic_index_metadata(document_id) == metadata_before

    provider.failing = False
    result = semantic.search(document_id, "阶段")
    assert result.matches
    assert result.matches[0].source_text == STAGES_TEXT


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(tmp_path: Path, query: str) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with pytest.raises(ValueError):
        semantic.search(document_id, query)


@pytest.mark.parametrize("top_k", [0, 21, True, 1.5])
def test_invalid_top_k_is_rejected(tmp_path: Path, top_k) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with pytest.raises((TypeError, ValueError)):
        semantic.search(document_id, "阶段", top_k=top_k)


@pytest.mark.parametrize("min_score", [float("nan"), float("inf"), -1.1, 1.1])
def test_invalid_min_score_is_rejected(tmp_path: Path, min_score: float) -> None:
    service, _ = _prepare(tmp_path)
    document_id = _seed_document(service, {0: STAGES_TEXT})
    semantic, _ = _build(service, FakeEmbeddingProvider(), document_id)

    with pytest.raises(ValueError):
        semantic.search(document_id, "阶段", min_score=min_score)
