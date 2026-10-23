"""Production PDF text search and geometry extraction."""

from __future__ import annotations

import math
from pathlib import Path

import pypdfium2.raw as pdfium_c
from pypdfium2 import PdfDocument, PdfiumError, PdfPage, PdfTextPage

from rag_anchor.domain import AnchorHit, PageDescriptor, PdfRect

_LINE_BREAK_CODEPOINTS = frozenset({0x0A, 0x0D})


class PdfAnchorError(RuntimeError):
    """Base class for PDF anchoring failures."""


class PdfOpenError(PdfAnchorError):
    """Raised when a PDF file exists but cannot be opened."""


class PdfReadError(PdfAnchorError):
    """Raised when an opened PDF cannot be read."""


def search_anchors(
    pdf_path: str | Path,
    query: str,
    *,
    match_case: bool = True,
) -> list[AnchorHit]:
    """Find every non-overlapping occurrence of ``query`` in ``pdf_path``.

    Parameters
    ----------
    pdf_path:
        Path to a local PDF file.
    query:
        Non-empty, non-whitespace search text. No whitespace normalization is
        applied.
    match_case:
        ``True`` (default) for case-sensitive matching; ``False`` for
        PDFium's case-insensitive matching.

    Returns
    -------
    list[AnchorHit]
        Hits sorted by ``(page_index, char_start)``. Returns ``[]`` when there
        is no match.

    Raises
    ------
    ValueError
        If ``query`` is empty or contains only whitespace.
    FileNotFoundError
        If ``pdf_path`` does not exist.
    IsADirectoryError
        If ``pdf_path`` is a directory.
    PdfOpenError
        If the file exists but PDFium cannot parse it.
    PdfReadError
        If the document opens but page text or geometry cannot be read.

    Notes
    -----
    Rectangles use absolute PDF user space points (lower-left origin, y up),
    exactly as returned by PDFium's ``get_charbox()``; the CropBox origin is
    not subtracted. Page rotation is reported separately and is never baked
    into the rectangles. Cross-line matches produce one rectangle per line
    rather than one rectangle spanning the inter-line gap.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string")

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"PDF path is not a file: {path}")

    try:
        document = PdfDocument(str(path))
    except PdfiumError as exc:
        raise PdfOpenError(f"Failed to open PDF: {path}") from exc

    try:
        return _search_document(document, query, match_case)
    except PdfiumError as exc:
        raise PdfReadError(f"Failed to read PDF: {path}") from exc
    finally:
        document.close()


def _search_document(
    document: PdfDocument,
    query: str,
    match_case: bool,
) -> list[AnchorHit]:
    hits: list[AnchorHit] = []
    for page_index in range(len(document)):
        page = document[page_index]
        text_page = page.get_textpage()
        try:
            hits.extend(_search_page(text_page, page, page_index, query, match_case))
        finally:
            text_page.close()

    hits.sort(key=lambda hit: (hit.page_index, hit.char_start))
    return hits


def _search_page(
    text_page: PdfTextPage,
    page: PdfPage,
    page_index: int,
    query: str,
    match_case: bool,
) -> list[AnchorHit]:
    descriptor = build_page_descriptor(page, page_index)

    searcher = text_page.search(query, match_case=match_case)
    page_hits: list[AnchorHit] = []
    try:
        while True:
            match = searcher.get_next()
            if match is None:
                break
            char_start, char_count = match
            page_hits.append(
                AnchorHit(
                    page_index=page_index,
                    page_number=page_index + 1,
                    matched_text=text_page.get_text_range(char_start, char_count),
                    char_start=char_start,
                    char_count=char_count,
                    rects=tuple(match_rects(text_page, char_start, char_count)),
                    page=descriptor,
                )
            )
    finally:
        searcher.close()

    return page_hits


def build_page_descriptor(page: PdfPage, page_index: int) -> PageDescriptor:
    """Describe one page's geometry for interpreting hit rectangles.

    ``width_pt`` / ``height_pt`` are the *unrotated* CropBox dimensions and
    ``cropbox`` is ``(left, bottom, right, top)`` in absolute PDF user space.
    Hit rectangles are also absolute user space, so a non-zero CropBox origin
    is not subtracted from them and the page rotation is reported separately
    instead of being baked into any rectangle.
    """
    cropbox_values = page.get_cropbox()
    cropbox: tuple[float, float, float, float] = (
        float(cropbox_values[0]),
        float(cropbox_values[1]),
        float(cropbox_values[2]),
        float(cropbox_values[3]),
    )
    return PageDescriptor(
        page_index=page_index,
        width_pt=cropbox[2] - cropbox[0],
        height_pt=cropbox[3] - cropbox[1],
        rotation=int(page.get_rotation()),
        cropbox=cropbox,
    )


def match_rects(
    text_page: PdfTextPage,
    char_start: int,
    char_count: int,
) -> list[PdfRect]:
    """Build one absolute PDF user space rectangle per matched text line.

    Both ``char_start`` and ``char_count`` are PDFium character indices, the
    same index space used by ``get_charbox()``.  Line breaks are detected from
    the PDFium character array rather than from a Python ``str`` slice:
    ``get_text_range()`` is UCS-2 only and may drop or re-encode characters, so
    Python string offsets must never be used as PDFium character indices.
    """
    rects: list[PdfRect] = []
    line_rects: list[PdfRect] = []

    def flush_line() -> None:
        if not line_rects:
            return
        rects.append(
            PdfRect(
                left=min(rect.left for rect in line_rects),
                bottom=min(rect.bottom for rect in line_rects),
                right=max(rect.right for rect in line_rects),
                top=max(rect.top for rect in line_rects),
            )
        )
        line_rects.clear()

    for index in range(char_start, char_start + char_count):
        if pdfium_c.FPDFText_GetUnicode(text_page, index) in _LINE_BREAK_CODEPOINTS:
            flush_line()
            continue

        left, bottom, right, top = text_page.get_charbox(index)
        if not all(math.isfinite(value) for value in (left, bottom, right, top)):
            continue
        if right <= left or top <= bottom:
            # PDFium emits degenerate boxes for some separators; a zero-area
            # box must not stretch the line rectangle.
            continue

        line_rects.append(
            PdfRect(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            )
        )

    flush_line()
    return rects
