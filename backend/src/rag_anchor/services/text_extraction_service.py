"""Page-level PDF text extraction with traceable source offsets.

This module performs *text extraction only*. It does not emit PDF coordinates,
embeddings, or chunks. The returned :class:`~rag_anchor.domain.text.PageText`
objects hold exactly the string produced by PDFium's ``get_text_range()`` for
each page, in page order.

Known engine limitation: ``get_text_range()`` is UCS-2 based. Supplementary
plane characters (for example U+1D11E) may be dropped or re-encoded by PDFium
before Python ever sees them. This module never claims otherwise, and because
chunk offsets are defined against the returned Python string, traceability
still holds for whatever text was actually extracted.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pypdfium2 import PdfDocument, PdfiumError

from rag_anchor.domain.text import PageText


class PdfTextExtractionError(RuntimeError):
    """Base class for PDF text extraction failures."""


class PdfTextOpenError(PdfTextExtractionError):
    """Raised when a PDF file exists but cannot be opened."""


class PdfTextReadError(PdfTextExtractionError):
    """Raised when an opened PDF cannot be read page by page."""


def normalize_document_id(document_id: UUID | str) -> str:
    """Return the canonical non-empty string form of a document identifier."""
    if isinstance(document_id, UUID):
        return str(document_id)
    if not isinstance(document_id, str):
        raise TypeError("document_id must be a UUID or str")
    normalized = document_id.strip()
    if not normalized:
        raise ValueError("document_id must be a non-empty string")
    return normalized


def extract_page_texts(
    pdf_path: str | Path,
    *,
    document_id: UUID | str,
) -> list[PageText]:
    """Extract one :class:`PageText` per page of ``pdf_path``.

    Parameters
    ----------
    pdf_path:
        Path to a local PDF file. The path is the project-managed file produced
        by the document service; no client-supplied path is trusted.
    document_id:
        Identifier recorded on every page. A :class:`~uuid.UUID` is stored in
        its canonical string form.

    Returns
    -------
    list[PageText]
        One entry per page, ordered by ``page_index`` (0-based). A page with no
        extractable text yields ``text == ""``; pages are never merged.

    Raises
    ------
    TypeError
        If ``document_id`` is not a ``UUID`` or ``str``.
    ValueError
        If ``document_id`` is empty or whitespace only.
    FileNotFoundError
        If ``pdf_path`` does not exist.
    IsADirectoryError
        If ``pdf_path`` is a directory.
    PdfTextOpenError
        If the file exists but PDFium cannot parse it.
    PdfTextReadError
        If the document opens but page text cannot be read.

    Notes
    -----
    Text is returned exactly as extracted. In particular, PDFium joins lines
    with ``\\r\\n``; this function does not normalize whitespace, strip
    punctuation, or otherwise modify the page text.
    """
    normalized_id = normalize_document_id(document_id)

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"PDF path is not a file: {path}")

    try:
        document = PdfDocument(str(path))
    except PdfiumError as exc:
        raise PdfTextOpenError(f"Failed to open PDF: {path}") from exc
    except OSError as exc:
        raise PdfTextOpenError(f"Failed to read PDF file: {path}") from exc

    try:
        return _extract_pages(document, normalized_id)
    except PdfiumError as exc:
        raise PdfTextReadError(f"Failed to read PDF text: {path}") from exc
    finally:
        document.close()


def _extract_pages(document: PdfDocument, document_id: str) -> list[PageText]:
    pages: list[PageText] = []
    for page_index in range(len(document)):
        page = document[page_index]
        text_page = page.get_textpage()
        try:
            text = text_page.get_text_range()
        finally:
            text_page.close()

        pages.append(
            PageText(
                document_id=document_id,
                page_index=page_index,
                page_number=page_index + 1,
                text=text,
            )
        )
    return pages
