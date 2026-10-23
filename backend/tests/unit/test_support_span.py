"""Unit tests for exact, conservative citation support spans."""

from __future__ import annotations

import pytest

from rag_anchor.services.support_span import (
    MAX_SUPPORT_TEXT_LENGTH,
    SUPPORT_KIND_RETRIEVAL_MATCH,
    SUPPORT_KIND_SENTENCE,
    build_support_span,
)


def _span(page: str, hit: str, *, occurrence: int = 0, chunk=None):
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = page.index(hit, cursor)
        cursor = start + 1
    end = start + len(hit)
    chunk_start, chunk_end = (0, len(page)) if chunk is None else chunk
    span = build_support_span(
        page,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        match_start=start,
        match_end=end,
    )
    assert page[span.start : span.end] == span.text
    assert chunk_start <= span.start <= start < end <= span.end <= chunk_end
    return span


def test_mid_sentence_hit_expands_to_the_complete_english_sentence() -> None:
    page = (
        "Intro sentence. The project has three stages: ingestion, retrieval, "
        "and evidence verification. Final note."
    )
    span = _span(page, "stages")

    assert span.kind == SUPPORT_KIND_SENTENCE
    assert span.text == (
        "The project has three stages: ingestion, retrieval, and evidence verification."
    )
    assert page[span.start : span.end] == span.text


def test_hit_in_the_first_sentence_starts_at_the_page_start() -> None:
    page = "The quick brown fox jumps over the lazy dog. Another sentence."
    span = _span(page, "quick")

    assert span.kind == SUPPORT_KIND_SENTENCE
    assert span.start == 0
    assert span.text == "The quick brown fox jumps over the lazy dog."


def test_repeated_identical_sentence_produces_distinct_offsets() -> None:
    page = "A. The project has stages. The project has stages. B."
    first = _span(page, "stages", occurrence=0)
    second = _span(page, "stages", occurrence=1)

    assert first.kind == second.kind == SUPPORT_KIND_SENTENCE
    assert first.text == second.text == "The project has stages."
    assert first.start != second.start
    assert page[first.start : first.end] == page[second.start : second.end]


@pytest.mark.parametrize("newline", ["\r\n", "\n", "\u2028"])
def test_cross_line_sentence_falls_back_to_the_hit(newline: str) -> None:
    page = f"The project has three{newline}stages: ingestion and retrieval."
    span = _span(page, "stages")

    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "stages"


def test_cjk_terminal_punctuation_selects_the_sentence() -> None:
    page = "项目包含三个阶段：采集、检索和证据校验。预算尚未确定。"  # noqa: RUF001
    span = _span(page, "证据校验")

    assert span.kind == SUPPORT_KIND_SENTENCE
    assert span.text == (
        "项目包含三个阶段：采集、检索和证据校验。"  # noqa: RUF001
    )


def test_cjk_closing_quote_is_included_in_the_sentence() -> None:
    page = "他说：“项目包含三个阶段。”然后离开。"  # noqa: RUF001
    span = _span(page, "项目")

    assert span.kind == SUPPORT_KIND_SENTENCE
    assert span.text == "他说：“项目包含三个阶段。”"  # noqa: RUF001


@pytest.mark.parametrize(
    ("page", "hit", "expected"),
    [
        ("Is it ready? Yes it is! Done.", "ready", "Is it ready?"),
        ("Is it ready? Yes it is! Done.", "Yes", "Yes it is!"),
        ('He said "Go." Then he left.', "Go", 'He said "Go."'),
    ],
)
def test_english_terminal_punctuation_and_closing_wrappers(
    page: str,
    hit: str,
    expected: str,
) -> None:
    span = _span(page, hit)

    assert span.kind == SUPPORT_KIND_SENTENCE
    assert span.text == expected


def test_decimal_point_is_not_a_sentence_boundary() -> None:
    page = "Pi is 3.14 and the hit lands here."
    span = _span(page, "hit")

    assert span.kind == SUPPORT_KIND_SENTENCE
    assert span.text == page


def test_hit_spanning_two_sentences_falls_back() -> None:
    page = "One sentence. Two sentence. Three."
    span = _span(page, "sentence. Two")

    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "sentence. Two"


def test_missing_terminal_punctuation_falls_back() -> None:
    page = "No punctuation at all with the hit here"
    span = _span(page, "hit")

    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "hit"


def test_chunk_boundary_before_the_sentence_start_falls_back() -> None:
    prefix = "Intro."
    page = prefix + " A sentence begins with stages and ends here."
    chunk_start = len(prefix) + 1
    span = _span(page, "stages", chunk=(chunk_start, len(page)))

    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "stages"


def test_chunk_boundary_after_the_hit_but_before_the_period_falls_back() -> None:
    page = "A sentence begins with stages and goes on. Next sentence."
    hit_start = page.index("stages")
    chunk_end = hit_start + len("stages") + 3
    span = _span(page, "stages", chunk=(0, chunk_end))

    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "stages"
    assert span.end <= chunk_end


def test_overlong_sentence_falls_back() -> None:
    page = ("x" * (MAX_SUPPORT_TEXT_LENGTH + 100)) + " hit end."
    span = _span(page, "hit")

    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "hit"


def test_chunk_start_is_not_assumed_to_be_a_sentence_boundary() -> None:
    page = "First sentence. The project has stages and ends here. Last sentence."
    chunk_start = page.index("The project")
    chunk_end = page.index("Last sentence")
    span = _span(page, "stages", chunk=(chunk_start, chunk_end))

    # Fixed-size chunking can start anywhere, so the absence of a terminator
    # before the hit is treated as an unknown sentence boundary.
    assert span.kind == SUPPORT_KIND_RETRIEVAL_MATCH
    assert span.text == "stages"
    assert chunk_start <= span.start < span.end <= chunk_end


@pytest.mark.parametrize(
    ("page", "chunk_start", "chunk_end", "match_start", "match_end"),
    [
        ("abc", 4, 3, 1, 2),
        ("abc", 0, 3, 3, 3),
        ("abc", 1, 2, 0, 1),
        ("abc", 0, 3, 2, 4),
    ],
)
def test_invalid_bounds_are_rejected(
    page: str,
    chunk_start: int,
    chunk_end: int,
    match_start: int,
    match_end: int,
) -> None:
    with pytest.raises(ValueError):
        build_support_span(
            page,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            match_start=match_start,
            match_end=match_end,
        )


def test_non_integer_bound_is_rejected() -> None:
    with pytest.raises(TypeError):
        build_support_span(
            "abc",
            chunk_start=0,
            chunk_end=3,
            match_start="1",  # type: ignore[arg-type]
            match_end=2,
        )
