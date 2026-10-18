"""Tests for the production ``search_anchors`` service."""

from __future__ import annotations

from pathlib import Path

import pypdfium2.raw as pdfium_c
import pytest
from pypdfium2 import PdfDocument

from rag_anchor.services import PdfOpenError, search_anchors

ABS = 0.05
_LINE_BREAK_CODEPOINTS = frozenset({0x0A, 0x0D})


def _raw_union(
    path: Path,
    page_index: int,
    char_start: int,
    char_count: int,
) -> tuple[float, float, float, float]:
    """Compute the absolute PDFium union for a character range."""
    with PdfDocument(str(path)) as document:
        text_page = document[page_index].get_textpage()
        text = text_page.get_text_range()
        boxes = [
            text_page.get_charbox(index)
            for index in range(char_start, char_start + char_count)
            if text[index] not in "\r\n"
        ]

    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _pdfium_union(
    path: Path,
    page_index: int,
    char_start: int,
    char_count: int,
) -> tuple[float, float, float, float]:
    """Compute a union using PDFium indices only.

    The regression guarded here is precisely the assumption that PDFium
    character indices and Python ``str`` offsets are interchangeable, so this
    helper must never index a Python string.
    """
    with PdfDocument(str(path)) as document:
        text_page = document[page_index].get_textpage()
        try:
            boxes = [
                text_page.get_charbox(index)
                for index in range(char_start, char_start + char_count)
                if pdfium_c.FPDFText_GetUnicode(text_page, index) not in _LINE_BREAK_CODEPOINTS
            ]
        finally:
            text_page.close()

    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def test_single_hit_returns_text_index_page_and_rect(single_page_pdf: Path) -> None:
    hits = search_anchors(single_page_pdf, "fox")

    assert len(hits) == 1
    hit = hits[0]
    assert hit.page_index == 0
    assert hit.page_number == 1
    assert hit.matched_text == "fox"
    assert hit.char_start == 16
    assert hit.char_count == 3
    assert len(hit.rects) == 1

    rect = hit.rects[0]
    assert rect.left == pytest.approx(191.536, abs=ABS)
    assert rect.bottom == pytest.approx(199.844, abs=ABS)
    assert rect.right == pytest.approx(207.256, abs=ABS)
    assert rect.top == pytest.approx(208.736, abs=ABS)
    assert hit.bbox == rect

    assert hit.page.page_index == 0
    assert hit.page.cropbox == (0.0, 0.0, 612.0, 792.0)
    assert hit.page.width_pt == 612.0
    assert hit.page.height_pt == 792.0
    assert hit.page.rotation == 0


def test_geometry_covers_drawn_text_and_matches_character_union(
    single_page_pdf: Path,
) -> None:
    hit = search_anchors(single_page_pdf, "brown fox")[0]

    assert hit.matched_text == "brown fox"
    assert hit.char_start == 10
    assert hit.char_count == 9

    raw_left, raw_bottom, raw_right, raw_top = _raw_union(
        single_page_pdf, hit.page_index, hit.char_start, hit.char_count
    )
    rect = hit.rects[0]

    assert rect.left == pytest.approx(raw_left, abs=1e-6)
    assert rect.bottom == pytest.approx(raw_bottom, abs=1e-6)
    assert rect.right == pytest.approx(raw_right, abs=1e-6)
    assert rect.top == pytest.approx(raw_top, abs=1e-6)

    # "The quick brown fox..." was drawn on one line with baseline y=200.
    assert rect.left >= 100.0
    assert rect.right <= 550.0
    assert rect.bottom <= 200.0 <= rect.top
    assert rect.height > 0.0


@pytest.mark.parametrize("query", ["zzzznotfound", "foxes"])
def test_zero_hits_return_empty_list(single_page_pdf: Path, query: str) -> None:
    assert search_anchors(single_page_pdf, query) == []
    assert search_anchors(single_page_pdf, query, match_case=False) == []


def test_same_page_repeats_are_all_returned_in_text_order(single_page_pdf: Path) -> None:
    hits = search_anchors(single_page_pdf, "quick")

    assert len(hits) == 3
    assert [hit.page_index for hit in hits] == [0, 0, 0]
    assert [hit.char_start for hit in hits] == [4, 46, 52]
    assert [hit.char_count for hit in hits] == [5, 5, 5]
    assert [hit.matched_text for hit in hits] == ["quick", "quick", "quick"]
    assert len({hit.rects[0].as_tuple() for hit in hits}) == 3


def test_match_case_controls_case_sensitivity(single_page_pdf: Path) -> None:
    sensitive = search_anchors(single_page_pdf, "quick", match_case=True)
    insensitive = search_anchors(single_page_pdf, "quick", match_case=False)

    assert [hit.matched_text for hit in sensitive] == ["quick", "quick", "quick"]
    assert [hit.char_start for hit in sensitive] == [4, 46, 52]

    assert len(insensitive) == 4
    assert [hit.matched_text for hit in insensitive] == [
        "quick",
        "quick",
        "quick",
        "QUICK",
    ]
    assert [hit.char_start for hit in insensitive] == [4, 46, 52, 58]


def test_multi_page_hits_use_zero_based_indices_and_sort_by_page(
    multi_page_pdf: Path,
) -> None:
    hits = search_anchors(multi_page_pdf, "gamma")

    assert len(hits) == 3
    assert [hit.page_index for hit in hits] == [0, 1, 2]
    assert [hit.page_number for hit in hits] == [1, 2, 3]
    assert [hit.char_start for hit in hits] == [18, 18, 18]
    assert [hit.matched_text for hit in hits] == ["gamma", "gamma", "gamma"]
    assert [hit.page.cropbox for hit in hits] == [(0.0, 0.0, 612.0, 792.0)] * 3


def test_cross_line_match_returns_one_rect_per_line(multi_page_pdf: Path) -> None:
    hits = search_anchors(multi_page_pdf, "gamma second")

    # The fixture draws the same two lines on all three pages, so every page
    # must produce its own cross-line hit.
    assert len(hits) == 3
    assert [hit.page_index for hit in hits] == [0, 1, 2]
    for hit in hits:
        assert hit.matched_text == "gamma\r\nsecond"
        assert hit.char_start == 18
        assert hit.char_count == 13
        assert len(hit.rects) == 2

    hit = hits[0]

    upper, lower = hit.rects
    assert upper.left == pytest.approx(201.220, abs=ABS)
    assert upper.bottom == pytest.approx(197.372, abs=ABS)
    assert upper.right == pytest.approx(240.436, abs=ABS)
    assert upper.top == pytest.approx(206.456, abs=ABS)

    assert lower.left == pytest.approx(100.384, abs=ABS)
    assert lower.bottom == pytest.approx(149.832, abs=ABS)
    assert lower.right == pytest.approx(138.004, abs=ABS)
    assert lower.top == pytest.approx(158.616, abs=ABS)

    # The two lines must stay separate; y grows upwards, so the first line is
    # above the second and no rectangle spans the inter-line gap.
    assert upper.bottom > lower.top
    assert upper.height < 12.0
    assert lower.height < 12.0

    bbox = hit.bbox
    assert bbox is not None
    assert bbox.bottom == pytest.approx(lower.bottom, abs=1e-6)
    assert bbox.top == pytest.approx(upper.top, abs=1e-6)
    assert bbox.height > upper.height + lower.height


def test_cropbox_keeps_absolute_user_space_coordinates(cropbox_pdf: Path) -> None:
    hit = search_anchors(cropbox_pdf, "quick")[0]
    rect = hit.rects[0]

    assert hit.page.cropbox == (50.0, 100.0, 400.0, 500.0)
    assert hit.page.width_pt == 350.0
    assert hit.page.height_pt == 400.0

    raw_left, raw_bottom, raw_right, raw_top = _raw_union(
        cropbox_pdf, hit.page_index, hit.char_start, hit.char_count
    )
    assert rect.left == pytest.approx(raw_left, abs=1e-6)
    assert rect.bottom == pytest.approx(raw_bottom, abs=1e-6)
    assert rect.right == pytest.approx(raw_right, abs=1e-6)
    assert rect.top == pytest.approx(raw_top, abs=1e-6)

    # M0-02 measured "quick" at absolute x=124.432, y=197.528 on this line
    # drawn with baseline (100, 200). The service must NOT return the
    # CropBox-relative values 74.432 / 97.528.
    assert rect.left == pytest.approx(124.432, abs=ABS)
    assert rect.bottom == pytest.approx(197.528, abs=ABS)
    assert rect.right == pytest.approx(152.032, abs=ABS)
    assert rect.top == pytest.approx(208.616, abs=ABS)

    # The drawn baseline is y=200 in absolute user space.
    assert rect.bottom <= 200.0 <= rect.top
    assert rect.left > hit.page.cropbox[0]
    assert rect.bottom > hit.page.cropbox[1]


def test_rotation_is_reported_but_not_baked_into_rects(pdf_factory) -> None:
    path = pdf_factory(
        "rotated.pdf",
        [[(100.0, 200.0, "The quick brown fox")]],
        rotation=90,
    )

    hit = search_anchors(path, "quick")[0]

    assert hit.page.rotation == 90
    # PageDescriptor keeps the *unrotated* CropBox dimensions.
    assert hit.page.width_pt == 612.0
    assert hit.page.height_pt == 792.0

    rect = hit.rects[0]
    assert rect.left == pytest.approx(124.432, abs=ABS)
    assert rect.bottom == pytest.approx(197.528, abs=ABS)
    assert rect.right == pytest.approx(152.032, abs=ABS)
    assert rect.top == pytest.approx(208.616, abs=ABS)


@pytest.mark.parametrize("query", ["", " ", "   ", "\n\t"])
def test_empty_or_whitespace_query_raises_value_error(
    single_page_pdf: Path,
    query: str,
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        search_anchors(single_page_pdf, query)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError, match=r"missing\.pdf"):
        search_anchors(missing, "anything")


def test_directory_path_raises_is_a_directory_error(tmp_path: Path) -> None:
    with pytest.raises(IsADirectoryError):
        search_anchors(tmp_path, "anything")


def test_corrupted_pdf_raises_pdf_open_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a valid pdf\n%%EOF")

    with pytest.raises(PdfOpenError, match="Failed to open PDF"):
        search_anchors(broken, "anything")


# --- Regression coverage A-F: PDFium character indices vs Python str offsets ---


def test_regression_a_plain_ascii_single_and_multiple_hits(single_page_pdf: Path) -> None:
    single = search_anchors(single_page_pdf, "fox")
    assert len(single) == 1
    assert single[0].char_start == 16
    assert single[0].matched_text == "fox"
    assert len(single[0].rects) == 1

    multiple = search_anchors(single_page_pdf, "quick")
    assert [hit.char_start for hit in multiple] == [4, 46, 52]
    assert [hit.matched_text for hit in multiple] == ["quick", "quick", "quick"]
    assert all(len(hit.rects) == 1 for hit in multiple)


def test_regression_b_line_break_match_keeps_one_rect_per_line(pdf_factory) -> None:
    path = pdf_factory(
        "line_break.pdf",
        [
            [
                (100.0, 200.0, "alpha beta"),
                (100.0, 150.0, "gamma delta"),
            ]
        ],
    )

    hit = search_anchors(path, "beta gamma")[0]

    assert hit.matched_text == "beta\r\ngamma"
    assert hit.char_start == 6
    assert hit.char_count == 11
    assert len(hit.rects) == 2

    upper, lower = hit.rects
    assert upper.bottom > lower.top

    expected_upper = _pdfium_union(path, hit.page_index, hit.char_start, 4)
    expected_lower = _pdfium_union(path, hit.page_index, hit.char_start + 6, 5)
    assert upper.as_tuple() == pytest.approx(expected_upper, abs=1e-6)
    assert lower.as_tuple() == pytest.approx(expected_lower, abs=1e-6)


@pytest.mark.parametrize(
    ("fixture_name", "expected_start"),
    [("accented_pdf", 5), ("cjk_pdf", 2)],
)
def test_regression_c_non_ascii_prefix_preserves_char_indices(
    request: pytest.FixtureRequest,
    fixture_name: str,
    expected_start: int,
) -> None:
    path: Path = request.getfixturevalue(fixture_name)

    hit = search_anchors(path, "quick")[0]

    assert hit.char_start == expected_start
    assert hit.char_count == 5
    assert hit.matched_text == "quick"
    assert len(hit.rects) == 1

    expected = _pdfium_union(path, hit.page_index, hit.char_start, hit.char_count)
    assert hit.rects[0].as_tuple() == pytest.approx(expected, abs=1e-6)


def test_regression_d_non_bmp_prefix_hit_uses_pdfium_indices(non_bmp_pdf: Path) -> None:
    with PdfDocument(str(non_bmp_pdf)) as document:
        text_page = document[0].get_textpage()
        try:
            # The regression being guarded: PDFium keeps the non-BMP character
            # in its character array, while get_text_range() drops it. Python
            # offsets therefore must never be used as PDFium indices.
            assert text_page.count_chars() == 6
            assert len(text_page.get_text_range()) == 5
        finally:
            text_page.close()

    hit = search_anchors(non_bmp_pdf, "quick")[0]

    assert hit.char_start == 1
    assert hit.char_count == 5
    assert hit.matched_text == "quick"
    expected = _pdfium_union(non_bmp_pdf, hit.page_index, hit.char_start, hit.char_count)
    assert hit.rects[0].as_tuple() == pytest.approx(expected, abs=1e-6)


def test_regression_d_unsupported_non_bmp_query_is_safe(non_bmp_pdf: Path) -> None:
    # Supplementary-plane characters are outside the documented support range.
    # The contract is a safe empty result, never an uncaught IndexError.
    assert search_anchors(non_bmp_pdf, "\U0001d11e") == []
    assert search_anchors(non_bmp_pdf, "\U0001d11equick") == []


def test_regression_e_rotation_and_cropbox_keep_absolute_coordinates(pdf_factory) -> None:
    path = pdf_factory(
        "rotated_cropbox.pdf",
        [[(100.0, 200.0, "The quick brown fox")]],
        rotation=90,
        cropbox=(50.0, 100.0, 400.0, 500.0),
    )

    hit = search_anchors(path, "quick")[0]

    assert hit.page.rotation == 90
    assert hit.page.cropbox == (50.0, 100.0, 400.0, 500.0)
    assert hit.page.width_pt == 350.0
    assert hit.page.height_pt == 400.0

    rect = hit.rects[0]
    assert rect.left == pytest.approx(124.432, abs=ABS)
    assert rect.bottom == pytest.approx(197.528, abs=ABS)
    assert rect.right == pytest.approx(152.032, abs=ABS)
    assert rect.top == pytest.approx(208.616, abs=ABS)
