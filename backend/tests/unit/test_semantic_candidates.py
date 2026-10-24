"""Offline tests for exactly traceable semantic candidates."""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_anchor.domain.text import PageText, TextChunk
from rag_anchor.services.semantic_candidates import (
    MAX_CANDIDATE_CHARS,
    build_candidate_id,
    build_semantic_candidates,
)


def _page(text: str, page_index: int = 0) -> PageText:
    return PageText(
        document_id=str(uuid4()),
        page_index=page_index,
        page_number=page_index + 1,
        text=text,
    )


def _chunk(
    page: PageText,
    *,
    start: int = 0,
    end: int | None = None,
    chunk_index: int = 0,
) -> TextChunk:
    resolved_end = len(page.text) if end is None else end
    return TextChunk(
        chunk_id=f"chk_{page.page_index}_{start}_{resolved_end}_{chunk_index}",
        document_id=page.document_id,
        page_index=page.page_index,
        page_number=page.page_number,
        start_offset=start,
        end_offset=resolved_end,
        text=page.text[start:resolved_end],
        chunk_index=chunk_index,
    )


def test_sentences_and_line_wraps_keep_exact_offsets() -> None:
    page = _page("First sentence. Second line\r\ncontinues here. 第三句。Final")
    chunk = _chunk(page)

    candidates = build_semantic_candidates([page], [chunk])

    assert [candidate.text for candidate in candidates] == [
        "First sentence.",
        "Second line\r\ncontinues here.",
        "第三句。",
        "Final",
    ]
    for candidate in candidates:
        assert page.text[candidate.start_offset : candidate.end_offset] == candidate.text
        assert chunk.start_offset <= candidate.start_offset
        assert candidate.end_offset <= chunk.end_offset
        assert candidate.chunk_id == chunk.chunk_id


def test_chunk_offsets_are_mapped_to_absolute_page_offsets() -> None:
    page = _page("HEADER First sentence. Second sentence.")
    chunk = _chunk(page, start=7)

    candidates = build_semantic_candidates([page], [chunk])

    assert [candidate.text for candidate in candidates] == [
        "First sentence.",
        "Second sentence.",
    ]
    for candidate in candidates:
        assert page.text[candidate.start_offset : candidate.end_offset] == candidate.text
        assert candidate.start_offset >= chunk.start_offset


def test_long_sentence_is_cut_into_windows() -> None:
    page = _page(("word " * 200).strip() + ".")
    chunk = _chunk(page)

    candidates = build_semantic_candidates([page], [chunk])

    assert len(candidates) >= 2
    assert all(len(candidate.text) <= MAX_CANDIDATE_CHARS for candidate in candidates)
    for candidate in candidates:
        assert page.text[candidate.start_offset : candidate.end_offset] == candidate.text
    assert "".join(candidate.text for candidate in candidates).replace(" ", "") == (
        page.text.replace(" ", "")
    )


def test_overlapping_chunks_emit_each_absolute_span_once() -> None:
    page = _page("Alpha sentence. Beta sentence.")
    first = _chunk(page, start=0, end=25, chunk_index=0)
    second = _chunk(page, start=10, end=len(page.text), chunk_index=1)

    candidates = build_semantic_candidates([page], [first, second])

    keys = [(candidate.start_offset, candidate.end_offset) for candidate in candidates]
    assert len(keys) == len(set(keys))
    alpha = next(candidate for candidate in candidates if candidate.text.startswith("Alpha"))
    assert alpha.chunk_id == first.chunk_id


def test_repeated_text_on_different_pages_keeps_distinct_offsets() -> None:
    first = _page("Repeated sentence.", page_index=0)
    second = _page("Repeated sentence.", page_index=1)

    candidates = build_semantic_candidates(
        [first, second],
        [_chunk(first), _chunk(second)],
    )

    assert len(candidates) == 2
    assert {candidate.page_index for candidate in candidates} == {0, 1}
    assert candidates[0].candidate_id != candidates[1].candidate_id


def test_candidate_id_is_deterministic() -> None:
    document_id = uuid4()
    first = build_candidate_id(
        document_id=document_id,
        page_index=0,
        start_offset=3,
        end_offset=9,
    )
    second = build_candidate_id(
        document_id=document_id,
        page_index=0,
        start_offset=3,
        end_offset=9,
    )
    other = build_candidate_id(
        document_id=document_id,
        page_index=0,
        start_offset=3,
        end_offset=10,
    )
    assert first == second
    assert first != other
    assert first.startswith("sem_")


def test_a_chunk_that_does_not_match_its_page_is_rejected() -> None:
    page = _page("The real page text.")
    chunk = TextChunk(
        chunk_id="chk_bad",
        document_id=page.document_id,
        page_index=0,
        page_number=1,
        start_offset=0,
        end_offset=len("Different text."),
        text="Different text.",
        chunk_index=0,
    )

    with pytest.raises(ValueError):
        build_semantic_candidates([page], [chunk])


def test_missing_page_is_rejected() -> None:
    page = _page("Some text.")
    chunk = _chunk(page)

    other_page = _page("Other text.", page_index=1)
    with pytest.raises(ValueError):
        build_semantic_candidates([other_page], [chunk])
