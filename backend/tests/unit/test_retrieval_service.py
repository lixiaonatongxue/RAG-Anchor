"""M1-02 tests: deterministic retrieval over persisted chunks.

The service is exercised against real SQLite rows produced by
``DocumentRepository.replace_document_index``. No PDF is parsed here: the
point of M1-02 is that retrieval reads the *persisted* index, not the source
document. PDF-level behaviour is covered by the API tests.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from rag_anchor.config import Settings
from rag_anchor.domain import PageText, TextChunk, build_chunk_id
from rag_anchor.infra import DocumentIndexRecord, DocumentRecord, DocumentRepository
from rag_anchor.services import (
    ChunkRetrievalService,
    DocumentNotFoundError,
    DocumentService,
)


def _prepare_service(tmp_path: Path, name: str = "data") -> DocumentService:
    """Create an initialized document service backed by a temporary database."""
    settings = Settings(data_dir=tmp_path / name)
    service = DocumentService(settings)
    service.initialize()
    return service


def _seed_document(
    service: DocumentService,
    *,
    pages: dict[int, str],
    chunks: list[tuple[int, int, int, int]],
    page_count: int | None = None,
) -> UUID:
    """Insert one document plus page text and chunk rows.

    ``chunks`` entries are ``(page_index, chunk_index, start_offset, end_offset)``.
    Chunk text is sliced from ``pages`` so the persistence invariant
    ``page.text[start:end] == chunk.text`` always holds.
    """
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
                page_index=page_index,
                start_offset=start_offset,
                end_offset=end_offset,
            ),
            document_id=document_key,
            page_index=page_index,
            page_number=page_index + 1,
            start_offset=start_offset,
            end_offset=end_offset,
            text=pages[page_index][start_offset:end_offset],
            chunk_index=chunk_index,
        )
        for page_index, chunk_index, start_offset, end_offset in chunks
    ]
    service.repository.create(
        DocumentRecord(
            document_id=document_id,
            filename="seed.pdf",
            size_bytes=1,
            page_count=page_count if page_count is not None else len(pages),
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
            max_chars=max((end - start for _, _, start, end in chunks), default=1),
            overlap_chars=0,
            fingerprint="fingerprint",
            page_count=len(pages),
            chunk_count=len(chunk_models),
            filtered_chunk_count=0,
            indexed_at=datetime.now(UTC),
        ),
    )
    return document_id


def test_hit_preserves_full_provenance(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    page_text = "alpha beta alpha gamma"
    document_id = _seed_document(
        service,
        pages={0: page_text},
        chunks=[(0, 0, 0, len(page_text))],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("beta")

    assert result.total == 1
    assert result.query == "beta"
    assert result.match_case is True
    assert result.document_id is None
    hit = result.matches[0]
    assert hit.document_id == document_id
    assert hit.page_index == 0
    assert hit.page_number == 1
    assert hit.chunk_index == 0
    assert hit.start_offset == 0
    assert hit.end_offset == len(page_text)
    assert hit.match_start == 6
    assert hit.match_end == 10
    assert hit.hit_text == "beta"
    assert hit.chunk_text == page_text
    assert page_text[hit.match_start : hit.match_end] == hit.hit_text
    assert hit.chunk_id.startswith("chk_")


def test_matching_is_case_sensitive_by_default(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    _seed_document(
        service,
        pages={0: "Alpha ALPHA alpha"},
        chunks=[(0, 0, 0, 17)],
    )
    retrieval = ChunkRetrievalService(service)

    assert retrieval.search_chunks("Alpha").total == 1
    assert retrieval.search_chunks("alpha").total == 1
    assert retrieval.search_chunks("ALPHA").total == 1


def test_case_insensitive_offsets_refer_to_original_text(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    page_text = "Alpha ALPHA alpha"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[(0, 0, 0, len(page_text))],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("alpha", match_case=False)

    assert result.total == 3
    assert result.match_case is False
    assert [hit.match_start for hit in result.matches] == [0, 6, 12]
    assert [hit.hit_text for hit in result.matches] == ["Alpha", "ALPHA", "alpha"]
    for hit in result.matches:
        assert page_text[hit.match_start : hit.match_end] == hit.hit_text


def test_matches_in_one_chunk_do_not_overlap(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    _seed_document(
        service,
        pages={0: "aaaaaa"},
        chunks=[(0, 0, 0, 6)],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("aa")

    assert [hit.match_start for hit in result.matches] == [0, 2, 4]


def test_no_results_returns_empty_page_with_zero_total(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    _seed_document(
        service,
        pages={0: "alpha beta"},
        chunks=[(0, 0, 0, 10)],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("missing")

    assert result.total == 0
    assert result.matches == []
    assert result.limit == 20
    assert result.offset == 0


def test_multi_document_search_and_document_filter(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    first = _seed_document(
        service,
        pages={0: "shared alpha one"},
        chunks=[(0, 0, 0, 16)],
    )
    second = _seed_document(
        service,
        pages={0: "shared alpha two"},
        chunks=[(0, 0, 0, 16)],
    )
    retrieval = ChunkRetrievalService(service)

    everything = retrieval.search_chunks("alpha")
    assert everything.total == 2
    assert {hit.document_id for hit in everything.matches} == {first, second}

    only_second = retrieval.search_chunks("alpha", document_id=second)
    assert only_second.total == 1
    assert only_second.matches[0].document_id == second


def test_results_are_deterministic_and_paginated(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    document_id = _seed_document(
        service,
        pages={0: "token token token"},
        chunks=[(0, 0, 0, 17)],
    )
    retrieval = ChunkRetrievalService(service)

    first_page = retrieval.search_chunks("token", limit=2, offset=0)
    second_page = retrieval.search_chunks("token", limit=2, offset=2)

    assert first_page.total == 3
    assert [hit.match_start for hit in first_page.matches] == [0, 6]
    assert second_page.total == 3
    assert [hit.match_start for hit in second_page.matches] == [12]
    assert {hit.document_id for hit in first_page.matches} == {document_id}

    repeated = retrieval.search_chunks("token", limit=2, offset=0)
    assert repeated.matches == first_page.matches


def test_exact_duplicate_span_across_overlapping_chunks_is_reported_once(
    tmp_path: Path,
) -> None:
    service = _prepare_service(tmp_path)
    page_text = "alpha beta alpha gamma"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[
            (0, 0, 0, 16),  # "alpha beta alpha"
            (0, 1, 11, 22),  # "alpha gamma"
        ],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("alpha")

    absolute_spans = [(hit.match_start, hit.match_end) for hit in result.matches]
    assert absolute_spans == [(0, 5), (11, 16)]
    assert len(set(absolute_spans)) == len(absolute_spans)
    # The lowest chunk index supplies the evidence for the shared span.
    assert [hit.chunk_index for hit in result.matches] == [0, 0]
    assert all(hit.chunk_id.startswith("chk_") for hit in result.matches)


def test_unknown_document_id_raises_not_found(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    retrieval = ChunkRetrievalService(service)

    with pytest.raises(DocumentNotFoundError):
        retrieval.search_chunks("alpha", document_id=uuid4())


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"query": ""}, ValueError),
        ({"query": "   "}, ValueError),
        ({"query": "x" * 2001}, ValueError),
        ({"query": "ok", "limit": 0}, ValueError),
        ({"query": "ok", "limit": 101}, ValueError),
        ({"query": "ok", "offset": -1}, ValueError),
        ({"query": 123}, TypeError),
        ({"query": "ok", "limit": True}, TypeError),
        ({"query": "ok", "offset": False}, TypeError),
    ],
)
def test_invalid_parameters_are_rejected(
    tmp_path: Path,
    kwargs: dict[str, object],
    exception: type[Exception],
) -> None:
    service = _prepare_service(tmp_path)
    retrieval = ChunkRetrievalService(service)

    with pytest.raises(exception):
        retrieval.search_chunks(**kwargs)  # type: ignore[arg-type]


def test_deleted_document_is_excluded_from_retrieval(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    keep = _seed_document(
        service,
        pages={0: "shared alpha keep"},
        chunks=[(0, 0, 0, 16)],
    )
    removed = _seed_document(
        service,
        pages={0: "shared alpha removed"},
        chunks=[(0, 0, 0, 19)],
    )
    retrieval = ChunkRetrievalService(service)
    assert retrieval.search_chunks("alpha").total == 2

    service.delete_document(removed)

    remaining = retrieval.search_chunks("alpha")
    assert remaining.total == 1
    assert remaining.matches[0].document_id == keep
    with pytest.raises(DocumentNotFoundError):
        retrieval.search_chunks("alpha", document_id=removed)
    assert service.repository.list_chunks(document_id=removed) == []


def test_restart_reads_persisted_chunks(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data")
    original = DocumentService(settings)
    original.initialize()
    document_id = _seed_document(
        original,
        pages={0: "restart safe alpha"},
        chunks=[(0, 0, 0, 18)],
    )

    restarted = DocumentService(Settings(data_dir=tmp_path / "data"))
    restarted.initialize()
    result = ChunkRetrievalService(restarted).search_chunks("alpha")

    assert result.total == 1
    assert result.matches[0].document_id == document_id
    assert result.matches[0].chunk_text == "restart safe alpha"


def test_list_chunks_is_ordered_and_scoped(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    document_id = _seed_document(
        service,
        pages={0: "page zero", 1: "page one"},
        chunks=[
            (1, 0, 0, 8),
            (0, 0, 0, 9),
            (1, 1, 5, 8),
        ],
    )

    chunks = service.repository.list_chunks(document_id=document_id)

    assert [(chunk.page_index, chunk.chunk_index) for chunk in chunks] == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)


def test_repository_listing_matches_index_rows(tmp_path: Path) -> None:
    """Guard against ordering drift in the SQLite query itself."""
    service = _prepare_service(tmp_path)
    document_id = _seed_document(
        service,
        pages={0: "alpha beta"},
        chunks=[(0, 0, 0, 10)],
    )
    repository = DocumentRepository(service.settings.database_path)

    indexed = service.repository.get_document_chunks(document_id)
    listed = repository.list_chunks(document_id=document_id)

    assert listed == indexed
    assert all(isinstance(chunk, TextChunk) for chunk in listed)


def test_uuid_type_is_preserved_in_hits(tmp_path: Path) -> None:
    service = _prepare_service(tmp_path)
    document_id = _seed_document(
        service,
        pages={0: "typed alpha"},
        chunks=[(0, 0, 0, 11)],
    )

    hit = ChunkRetrievalService(service).search_chunks("alpha").matches[0]

    assert isinstance(hit.document_id, UUID)
    assert hit.document_id == document_id


@pytest.mark.parametrize(
    ("query", "match_case"),
    [
        ("aa", True),
        ("AA", False),
    ],
)
def test_repeated_character_query_is_selected_left_to_right(
    tmp_path: Path,
    query: str,
    match_case: bool,
) -> None:
    """``aa`` in ``aaaa`` must yield ``[0, 2)`` and ``[2, 4)``, not ``[1, 3)``.

    Candidates are collected for every position (0, 1, 2) and only afterwards
    filtered for non-overlap. Reserving ``[0, 2)`` first forces the next hit to
    start at 2 instead of the currently-skipped ``[1, 3)``.
    """
    service = _prepare_service(tmp_path)
    page_text = "aaaa"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[(0, 0, 0, len(page_text))],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks(query, match_case=match_case)

    assert result.total == 2
    assert [(hit.match_start, hit.match_end) for hit in result.matches] == [(0, 2), (2, 4)]
    assert [hit.hit_text for hit in result.matches] == ["aa", "aa"]
    for hit in result.matches:
        assert page_text[hit.match_start : hit.match_end] == hit.hit_text


def test_overlapping_chunks_keep_later_distinct_page_hits(tmp_path: Path) -> None:
    """Page-level filtering must not inherit a chunk-local cursor.

    Chunk 0 covers ``[0, 3)`` and chunk 1 covers ``[1, 4)``. A chunk-local
    non-overlap cursor would select absolute ``[0, 2)`` and ``[1, 3)``; the
    page-level pass instead selects ``[0, 2)`` and ``[2, 4)``, with the second
    hit attributed to the only chunk that fully contains it.
    """
    service = _prepare_service(tmp_path)
    page_text = "aaaa"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[
            (0, 0, 0, 3),
            (0, 1, 1, 4),
        ],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("aa")

    assert result.total == 2
    assert [(hit.match_start, hit.match_end) for hit in result.matches] == [(0, 2), (2, 4)]
    assert [hit.chunk_index for hit in result.matches] == [0, 1]
    assert [hit.chunk_text for hit in result.matches] == ["aaa", "aaa"]
    for hit in result.matches:
        assert page_text[hit.match_start : hit.match_end] == hit.hit_text == "aa"


def test_multiple_distinct_hits_survive_cross_chunk_deduplication(tmp_path: Path) -> None:
    """Distinct page hits survive while duplicate spans are collapsed."""
    service = _prepare_service(tmp_path)
    page_text = "aa aa aa"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[
            (0, 0, 0, 5),  # "aa aa"
            (0, 1, 3, 8),  # "aa aa"
        ],
    )
    retrieval = ChunkRetrievalService(service)

    result = retrieval.search_chunks("aa")

    assert result.total == 3
    assert [(hit.match_start, hit.match_end) for hit in result.matches] == [
        (0, 2),
        (3, 5),
        (6, 8),
    ]
    # The shared span (3, 5) is reported once, from the lowest chunk index.
    assert [hit.chunk_index for hit in result.matches] == [0, 0, 1]
    assert all(page_text[hit.match_start : hit.match_end] == hit.hit_text for hit in result.matches)


def test_cross_chunk_boundary_query_is_not_stitched(tmp_path: Path) -> None:
    """Documented boundary: chunks are never concatenated for matching.

    Page ``abcdefgh`` is stored as ``abcd`` + ``efgh``. ``defg`` only exists
    when the two chunks are joined, so it must return nothing. Queries fully
    contained in one chunk continue to match with page-original offsets.
    """
    service = _prepare_service(tmp_path)
    page_text = "abcdefgh"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[
            (0, 0, 0, 4),
            (0, 1, 4, 8),
        ],
    )
    retrieval = ChunkRetrievalService(service)

    assert retrieval.search_chunks("defg").total == 0
    assert retrieval.search_chunks("cde").total == 0
    assert retrieval.search_chunks("efg").total == 1

    inside_first = retrieval.search_chunks("cd")
    assert inside_first.total == 1
    assert (inside_first.matches[0].match_start, inside_first.matches[0].match_end) == (2, 4)
    assert inside_first.matches[0].chunk_index == 0

    inside_second = retrieval.search_chunks("ef")
    assert inside_second.total == 1
    assert (inside_second.matches[0].match_start, inside_second.matches[0].match_end) == (4, 6)
    assert inside_second.matches[0].chunk_index == 1


def test_case_insensitive_matching_does_not_equate_sharp_s_with_ss(tmp_path: Path) -> None:
    """Documented boundary: no length-changing Unicode case folding.

    ``str.lower()`` is applied to same-length windows only, so ``ß`` and ``SS``
    are not equivalent even with ``match_case=False``. Offsets stay aligned to
    the original page text.
    """
    service = _prepare_service(tmp_path)
    page_text = "straße"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[(0, 0, 0, len(page_text))],
    )
    retrieval = ChunkRetrievalService(service)

    sharp_s = retrieval.search_chunks("ß", match_case=False)
    assert sharp_s.total == 1
    assert sharp_s.matches[0].match_start == 4
    assert page_text[sharp_s.matches[0].match_start : sharp_s.matches[0].match_end] == "ß"

    assert retrieval.search_chunks("SS", match_case=False).total == 0
    assert retrieval.search_chunks("ss", match_case=False).total == 0
    assert retrieval.search_chunks("STRASSE", match_case=False).total == 0
    # The literal character still matches when case sensitivity is requested.
    assert retrieval.search_chunks("ß").total == 1


def test_pagination_counts_page_level_non_overlapping_hits(tmp_path: Path) -> None:
    """``total`` is computed after de-duplication and overlap filtering."""
    service = _prepare_service(tmp_path)
    page_text = "aaaa"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[(0, 0, 0, len(page_text))],
    )
    retrieval = ChunkRetrievalService(service)

    first = retrieval.search_chunks("aa", limit=1, offset=0)
    second = retrieval.search_chunks("aa", limit=1, offset=1)
    past_end = retrieval.search_chunks("aa", limit=1, offset=2)

    assert first.total == 2
    assert [(hit.match_start, hit.match_end) for hit in first.matches] == [(0, 2)]
    assert second.total == 2
    assert [(hit.match_start, hit.match_end) for hit in second.matches] == [(2, 4)]
    assert past_end.total == 2
    assert past_end.matches == []


def test_repeated_overlapping_search_is_deterministic(tmp_path: Path) -> None:
    """Order and evidence attribution are reproducible across calls."""
    service = _prepare_service(tmp_path)
    page_text = "aa aa aa"
    _seed_document(
        service,
        pages={0: page_text},
        chunks=[
            (0, 0, 0, 5),
            (0, 1, 3, 8),
        ],
    )
    retrieval = ChunkRetrievalService(service)

    first = retrieval.search_chunks("aa")
    second = retrieval.search_chunks("aa")

    first_evidence = [
        (hit.match_start, hit.match_end, hit.chunk_index, hit.chunk_id) for hit in first.matches
    ]
    second_evidence = [
        (hit.match_start, hit.match_end, hit.chunk_index, hit.chunk_id) for hit in second.matches
    ]

    assert first.matches == second.matches
    assert first_evidence == second_evidence
