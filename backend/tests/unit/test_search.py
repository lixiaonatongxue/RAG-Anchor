"""Verify pypdfium2 text search semantics used by ``search_anchors``."""

from __future__ import annotations

from pathlib import Path

from pypdfium2 import PdfDocument, PdfTextPage


def _open(path: Path) -> PdfDocument:
    return PdfDocument(str(path))


def _all_hits(text_page: PdfTextPage, query: str, **kwargs: object) -> list[tuple[int, int]]:
    """Drain ``PdfTextSearcher`` and return every ``(start_index, count)`` hit."""
    searcher = text_page.search(query, **kwargs)  # type: ignore[arg-type]
    hits: list[tuple[int, int]] = []
    while True:
        hit = searcher.get_next()
        if hit is None:
            return hits
        hits.append(hit)


def test_search_returns_character_index_and_count(single_page_pdf: Path) -> None:
    document = _open(single_page_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        assert _all_hits(text_page, "fox") == [(16, 3)]

        start, count = _all_hits(text_page, "fox")[0]
        assert text[start : start + count] == "fox"
        assert text_page.get_text_range(start, count) == "fox"
    finally:
        document.close()


def test_search_finds_all_occurrences_and_is_case_insensitive_by_default(
    single_page_pdf: Path,
) -> None:
    document = _open(single_page_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        hits = _all_hits(text_page, "quick")

        assert len(hits) == 4
        assert hits[0] == (4, 5)
        for start, count in hits:
            assert text[start : start + count].casefold() == "quick"

        case_sensitive_hits = _all_hits(text_page, "quick", match_case=True)

        assert len(case_sensitive_hits) == 3
        for start, count in case_sensitive_hits:
            assert text[start : start + count] == "quick"
    finally:
        document.close()


def test_search_returns_no_hits_for_absent_text(single_page_pdf: Path) -> None:
    document = _open(single_page_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()

        assert _all_hits(text_page, "zzzznotfound") == []
    finally:
        document.close()


def test_search_matches_phrase_across_line_break(multi_page_pdf: Path) -> None:
    document = _open(multi_page_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        assert "\r\n" in text

        hits = _all_hits(text_page, "gamma second")

        assert len(hits) == 1
        start, count = hits[0]
        assert text[start : start + count] == "gamma\r\nsecond"
    finally:
        document.close()


def test_character_count_matches_extracted_text(multi_page_pdf: Path) -> None:
    document = _open(multi_page_pdf)
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            text_page = page.get_textpage()

            assert text_page.count_chars() == len(text_page.get_text_range())
    finally:
        document.close()


def test_pages_are_addressed_by_zero_based_index(multi_page_pdf: Path) -> None:
    document = _open(multi_page_pdf)
    try:
        assert len(document) == 3

        for page_index in range(len(document)):
            page = document[page_index]
            text_page = page.get_textpage()
            text = text_page.get_text_range()

            assert f"Page {page_index + 1} alpha beta gamma" in text
            assert len(_all_hits(text_page, f"Page {page_index + 1}")) == 1
    finally:
        document.close()
