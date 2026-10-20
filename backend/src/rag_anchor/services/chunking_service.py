"""Deterministic, lossless chunk splitting over extracted page text.

The first version deliberately splits on character count only. It does not use
a tokenizer, embeddings, or any retrieval framework. Every chunk is an exact
Python string slice of one page, never a normalised or rewritten copy.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_anchor.domain.text import (
    TEXT_EXTRACTION_VERSION,
    PageText,
    TextChunk,
    build_chunk_id,
)

DEFAULT_MAX_CHARS = 800
"""Default maximum chunk length in Python characters."""

DEFAULT_OVERLAP_CHARS = 0
"""Default overlap between consecutive chunks in Python characters."""


def chunk_page_texts(
    pages: Sequence[PageText],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    extraction_version: str = TEXT_EXTRACTION_VERSION,
) -> list[TextChunk]:
    """Split every page into deterministic, traceable chunks.

    Each page is processed independently: the returned chunks never span two
    pages and each chunk records its own ``page_index`` / ``page_number``.
    Chunks are emitted in ascending page order and, within a page, in ascending
    ``start_offset`` order.

    Parameters
    ----------
    pages:
        Page texts belonging to one document. Input order is irrelevant; the
        pages are sorted by ``page_index``. Duplicate page indices or mixed
        document IDs are rejected.
    max_chars:
        Maximum length of one chunk in Python characters. Must be positive.
    overlap_chars:
        Number of characters shared by two consecutive chunks on the same page.
        Must satisfy ``0 <= overlap_chars < max_chars``.
    extraction_version:
        Version tag recorded on each chunk and included in its ID.

    Returns
    -------
    list[TextChunk]
        Possibly empty. Pages whose text is exactly ``""`` produce no chunks; a
        whitespace-only page is not empty and is preserved verbatim. The final
        chunk of a page may be shorter than ``max_chars`` and is always emitted.

    Raises
    ------
    TypeError
        If a numeric parameter is not an integer.
    ValueError
        If ``max_chars <= 0``, if ``overlap_chars < 0``, if
        ``overlap_chars >= max_chars``, if ``pages`` mixes documents, or if
        ``pages`` contains duplicate page indices.
    """
    _require_int("max_chars", max_chars)
    _require_int("overlap_chars", overlap_chars)
    if max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be zero or greater")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    if not isinstance(extraction_version, str) or not extraction_version:
        raise ValueError("extraction_version must be a non-empty string")

    ordered = _ordered_pages(pages)
    step = max_chars - overlap_chars
    chunks: list[TextChunk] = []

    for page in ordered:
        text = page.text
        if not text:
            continue

        page_chunk_index = 0
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(
                TextChunk(
                    chunk_id=build_chunk_id(
                        document_id=page.document_id,
                        page_index=page.page_index,
                        start_offset=start,
                        end_offset=end,
                        extraction_version=extraction_version,
                    ),
                    document_id=page.document_id,
                    page_index=page.page_index,
                    page_number=page.page_number,
                    start_offset=start,
                    end_offset=end,
                    text=text[start:end],
                    chunk_index=page_chunk_index,
                    extraction_version=extraction_version,
                )
            )
            page_chunk_index += 1
            if end >= len(text):
                break
            start += step

    return chunks


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")


def _ordered_pages(pages: Sequence[PageText]) -> list[PageText]:
    ordered = sorted(pages, key=lambda page: page.page_index)
    previous: PageText | None = None
    for page in ordered:
        if previous is not None:
            if page.page_index == previous.page_index:
                raise ValueError("pages must not contain duplicate page_index values")
            if page.document_id != previous.document_id:
                raise ValueError("all pages must belong to the same document_id")
        previous = page
    return ordered
