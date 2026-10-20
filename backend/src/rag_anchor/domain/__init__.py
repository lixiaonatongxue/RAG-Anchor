"""Domain models for PDF evidence anchoring."""

from rag_anchor.domain.models import AnchorHit, PageDescriptor, PdfRect
from rag_anchor.domain.text import (
    TEXT_EXTRACTION_VERSION,
    PageText,
    TextChunk,
    build_chunk_id,
)

__all__ = [
    "TEXT_EXTRACTION_VERSION",
    "AnchorHit",
    "PageDescriptor",
    "PageText",
    "PdfRect",
    "TextChunk",
    "build_chunk_id",
]
