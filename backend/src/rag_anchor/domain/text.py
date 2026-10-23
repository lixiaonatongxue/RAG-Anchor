"""Traceable page text and retrieval chunk models.

Index contract
--------------

``PageText.text`` is exactly the string returned by PDFium's
``get_text_range()`` for one page. Pages are never concatenated.

``TextChunk.start_offset`` / ``end_offset`` are Python ``str`` slice boundaries
into the *stored* ``PageText.text`` string. They are:

* **not** PDFium native character indices,
* **not** UTF-16 code unit offsets,
* **not** PDF page coordinates.

The traceability invariant is::

    page.text[chunk.start_offset:chunk.end_offset] == chunk.text
A chunk is not a verified PDF anchor. Locating a chunk (or any page-text span)
in the rendered PDF must go through
:class:`rag_anchor.services.evidence_service.EvidenceLocationService`
(the ``POST /api/v1/documents/{document_id}/evidence/locate`` endpoint) or the
existing anchor search API, and only ever accept rectangles that were actually
returned by PDFium. Python ``str`` offsets are never PDFium character indices:
the evidence location service maps between the two spaces explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

TEXT_EXTRACTION_VERSION = "pdfium-text-v1"
"""Version tag describing the semantics of the extracted page text."""

_CHUNK_ID_PREFIX = "chk_"
_CHUNK_ID_HEX_LENGTH = 32


@dataclass(frozen=True, slots=True)
class PageText:
    """Text extracted from one PDF page.

    ``page_index`` is zero-based and ``page_number`` is one-based; the model
    rejects any other relationship so callers can never mix the two.
    """

    document_id: str
    page_index: int
    page_number: int
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int):
            raise TypeError("page_index must be an integer")
        if self.page_index < 0:
            raise ValueError("page_index must be zero or greater")
        if self.page_number != self.page_index + 1:
            raise ValueError("page_number must equal page_index + 1")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A lossless slice of one :class:`PageText`.

    ``chunk_index`` is the zero-based ordinal of the chunk *within its page*.
    ``extraction_version`` records the extraction semantics the offsets were
    produced against.
    """

    chunk_id: str
    document_id: str
    page_index: int
    page_number: int
    start_offset: int
    end_offset: int
    text: str
    chunk_index: int = 0
    extraction_version: str = TEXT_EXTRACTION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if not isinstance(self.document_id, str) or not self.document_id:
            raise ValueError("document_id must be a non-empty string")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int):
            raise TypeError("page_index must be an integer")
        if self.page_index < 0:
            raise ValueError("page_index must be zero or greater")
        if self.page_number != self.page_index + 1:
            raise ValueError("page_number must equal page_index + 1")
        if isinstance(self.start_offset, bool) or not isinstance(self.start_offset, int):
            raise TypeError("start_offset must be an integer")
        if isinstance(self.end_offset, bool) or not isinstance(self.end_offset, int):
            raise TypeError("end_offset must be an integer")
        if self.start_offset < 0:
            raise ValueError("start_offset must be zero or greater")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be greater than or equal to start_offset")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if len(self.text) != self.end_offset - self.start_offset:
            raise ValueError("text length must equal end_offset - start_offset")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int):
            raise TypeError("chunk_index must be an integer")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be zero or greater")
        if not isinstance(self.extraction_version, str) or not self.extraction_version:
            raise ValueError("extraction_version must be a non-empty string")

    def verify_source(self, page: PageText) -> None:
        """Raise ``ValueError`` unless this chunk is an exact slice of ``page``."""
        if self.document_id != page.document_id:
            raise ValueError("chunk document_id does not match the source page")
        if self.page_index != page.page_index:
            raise ValueError("chunk page_index does not match the source page")
        if page.text[self.start_offset : self.end_offset] != self.text:
            raise ValueError("chunk text is not the exact slice of the source page")


def build_chunk_id(
    *,
    document_id: str,
    page_index: int,
    start_offset: int,
    end_offset: int,
    extraction_version: str = TEXT_EXTRACTION_VERSION,
) -> str:
    """Return a deterministic chunk identifier.

    The identifier is a truncated SHA-256 digest over a canonical JSON tuple.
    Python's built-in ``hash()`` is never used because it is salted per process
    and therefore not stable across runs or machines.
    """
    payload = json.dumps(
        [extraction_version, document_id, page_index, start_offset, end_offset],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_CHUNK_ID_PREFIX}{digest[:_CHUNK_ID_HEX_LENGTH]}"
