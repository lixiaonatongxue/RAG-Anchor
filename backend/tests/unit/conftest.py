"""Runtime-generated PDF fixtures for the M0-02 PDF engine spike.

No binary PDF is stored in the repository. Every sample is written to pytest's
``tmp_path`` so that the text baselines, rotation and crop box used by the
geometry assertions stay explicit and reviewable.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeAlias

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas

Line: TypeAlias = tuple[float, float, str]
PageLines: TypeAlias = Sequence[Line]
CropBox: TypeAlias = tuple[float, float, float, float]

DEFAULT_PAGE_SIZE: tuple[float, float] = (612.0, 792.0)
FONT_NAME = "Helvetica"
FONT_SIZE = 12


def write_pdf(
    path: Path,
    pages: Iterable[PageLines],
    *,
    page_size: tuple[float, float] = DEFAULT_PAGE_SIZE,
    rotation: int = 0,
    cropbox: CropBox | None = None,
) -> Path:
    """Write a PDF with known text baselines and optional rotation/crop box."""
    buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=page_size)
    for lines in pages:
        pdf_canvas.setFont(FONT_NAME, FONT_SIZE)
        for x, y, text in lines:
            pdf_canvas.drawString(x, y, text)
        pdf_canvas.showPage()
    pdf_canvas.save()
    buffer.seek(0)

    reader = PdfReader(buffer)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    for page in writer.pages:
        if rotation:
            page.rotate(rotation)
        if cropbox is not None:
            page.cropbox = RectangleObject([float(value) for value in cropbox])

    output = io.BytesIO()
    writer.write(output)
    path.write_bytes(output.getvalue())
    return path


@pytest.fixture
def pdf_factory(tmp_path: Path):
    """Return a helper that writes a generated PDF into ``tmp_path``."""

    def build(name: str, pages: Iterable[PageLines], **kwargs: object) -> Path:
        return write_pdf(tmp_path / name, pages, **kwargs)  # type: ignore[arg-type]

    return build


@pytest.fixture
def single_page_pdf(pdf_factory) -> Path:
    return pdf_factory(
        "single.pdf",
        [
            [
                (100.0, 200.0, "The quick brown fox jumps over the lazy dog."),
                (100.0, 300.0, "quick quick QUICK"),
            ]
        ],
    )


@pytest.fixture
def multi_page_pdf(pdf_factory) -> Path:
    return pdf_factory(
        "multi.pdf",
        [
            [
                (100.0, 200.0, "Page 1 alpha beta gamma"),
                (100.0, 150.0, "second line quick"),
            ],
            [
                (100.0, 200.0, "Page 2 alpha beta gamma"),
                (100.0, 150.0, "second line quick"),
            ],
            [
                (100.0, 200.0, "Page 3 alpha beta gamma"),
                (100.0, 150.0, "second line quick"),
            ],
        ],
    )


@pytest.fixture
def cropbox_pdf(pdf_factory) -> Path:
    return pdf_factory(
        "cropbox.pdf",
        [[(100.0, 200.0, "The quick brown fox jumps over the lazy dog.")]],
        cropbox=(50.0, 100.0, 400.0, 500.0),
    )
