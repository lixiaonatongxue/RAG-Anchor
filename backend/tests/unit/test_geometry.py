"""Verify pypdfium2 page geometry in PDF user space.

Observed semantics on pypdfium2 5.14.0 / PDFium (documented in ADR-0001):

* ``get_charbox()`` returns absolute PDF user-space coordinates: lower-left
  origin, y grows upwards.
* ``get_charbox()`` is *not* shifted by a non-default CropBox.
* ``get_rotation()`` reports clockwise page rotation in degrees.
* ``get_size()`` reports the displayed size after applying rotation and CropBox.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdfium2 import PdfDocument


def _open(path: Path) -> PdfDocument:
    return PdfDocument(str(path))


def test_charbox_uses_pdf_user_space_with_lower_left_origin(single_page_pdf: Path) -> None:
    document = _open(single_page_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        index = text.index("The")
        left, bottom, right, top = text_page.get_charbox(index)

        # reportlab drew "The..." with baseline (100, 200), font size 12.
        assert left == pytest.approx(100.168, abs=0.05)
        assert bottom == pytest.approx(200.0, abs=0.05)
        assert right == pytest.approx(107.164, abs=0.05)
        assert top == pytest.approx(208.616, abs=0.05)
        assert left < right
        assert bottom < top
    finally:
        document.close()


def test_merging_character_boxes_yields_word_bounding_box(single_page_pdf: Path) -> None:
    document = _open(single_page_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        start = text.index("The")
        boxes = [text_page.get_charbox(index) for index in range(start, start + len("The"))]

        left = min(box[0] for box in boxes)
        bottom = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        top = max(box[3] for box in boxes)

        assert left == pytest.approx(100.168, abs=0.05)
        assert bottom == pytest.approx(199.832, abs=0.05)
        assert right == pytest.approx(120.196, abs=0.05)
        assert top == pytest.approx(208.616, abs=0.05)
    finally:
        document.close()


def test_cropbox_sets_page_size_but_charbox_stays_absolute(cropbox_pdf: Path) -> None:
    document = _open(cropbox_pdf)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        assert page.get_cropbox() == pytest.approx((50.0, 100.0, 400.0, 500.0))
        assert page.get_size() == pytest.approx((350.0, 400.0))

        index = text.index("quick")
        left, bottom, _, _ = text_page.get_charbox(index)

        # PDFium reports absolute user space, not CropBox-relative coordinates.
        assert left == pytest.approx(124.432, abs=0.05)
        assert bottom == pytest.approx(197.528, abs=0.05)

        crop_left, crop_bottom, _, _ = page.get_cropbox()

        # Explicit conversion the production adapter must perform.
        assert left - crop_left == pytest.approx(74.432, abs=0.05)
        assert bottom - crop_bottom == pytest.approx(97.528, abs=0.05)
    finally:
        document.close()


@pytest.mark.parametrize(
    ("rotation", "expected_size"),
    [
        (90, (792.0, 612.0)),
        (180, (612.0, 792.0)),
        (270, (792.0, 612.0)),
    ],
)
def test_rotation_is_reported_without_rotating_charboxes(
    pdf_factory,
    rotation: int,
    expected_size: tuple[float, float],
) -> None:
    path = pdf_factory(
        f"rotated-{rotation}.pdf",
        [[(100.0, 200.0, "The quick brown fox")]],
        rotation=rotation,
    )
    document = _open(path)
    try:
        page = document[0]
        text_page = page.get_textpage()
        text = text_page.get_text_range()

        assert page.get_rotation() == rotation
        assert page.get_size() == pytest.approx(expected_size)
        assert page.get_cropbox() == pytest.approx((0.0, 0.0, 612.0, 792.0))
        assert "quick" in text

        index = text.index("quick")
        left, bottom, _, _ = text_page.get_charbox(index)

        # Coordinates are still unrotated absolute user space.
        assert left == pytest.approx(124.432, abs=0.05)
        assert bottom == pytest.approx(197.528, abs=0.05)
    finally:
        document.close()


def test_rotation_and_cropbox_combine_in_reported_page_size(pdf_factory) -> None:
    path = pdf_factory(
        "rotated-cropbox.pdf",
        [[(100.0, 200.0, "The quick brown fox")]],
        rotation=90,
        cropbox=(50.0, 100.0, 400.0, 500.0),
    )
    document = _open(path)
    try:
        page = document[0]

        assert page.get_rotation() == 90
        assert page.get_cropbox() == pytest.approx((50.0, 100.0, 400.0, 500.0))
        # CropBox is 350 x 400; a quarter turn swaps the displayed dimensions.
        assert page.get_size() == pytest.approx((400.0, 350.0))
    finally:
        document.close()
