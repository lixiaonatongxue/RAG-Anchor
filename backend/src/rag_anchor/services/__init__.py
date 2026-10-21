"""Application services."""

from rag_anchor.services.anchor_service import (
    PdfAnchorError,
    PdfOpenError,
    PdfReadError,
    search_anchors,
)
from rag_anchor.services.chunking_service import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_page_texts,
)
from rag_anchor.services.document_service import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentService,
    DocumentServiceError,
    InvalidUploadError,
    UploadTooLargeError,
)
from rag_anchor.services.index_service import (
    DocumentIndexError,
    DocumentIndexPersistenceError,
    DocumentIndexService,
    DocumentIndexSourceError,
    IndexResult,
)
from rag_anchor.services.retrieval_service import (
    DEFAULT_RETRIEVAL_LIMIT,
    MAX_QUERY_LENGTH,
    MAX_RETRIEVAL_LIMIT,
    ChunkRetrievalService,
    RetrievalMatch,
    RetrievalResult,
)
from rag_anchor.services.text_extraction_service import (
    PdfTextExtractionError,
    PdfTextOpenError,
    PdfTextReadError,
    extract_page_texts,
    normalize_document_id,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_OVERLAP_CHARS",
    "DEFAULT_RETRIEVAL_LIMIT",
    "MAX_QUERY_LENGTH",
    "MAX_RETRIEVAL_LIMIT",
    "ChunkRetrievalService",
    "DocumentIndexError",
    "DocumentIndexPersistenceError",
    "DocumentIndexService",
    "DocumentIndexSourceError",
    "DocumentNotFoundError",
    "DocumentPersistenceError",
    "DocumentService",
    "DocumentServiceError",
    "IndexResult",
    "InvalidUploadError",
    "PdfAnchorError",
    "PdfOpenError",
    "PdfReadError",
    "PdfTextExtractionError",
    "PdfTextOpenError",
    "PdfTextReadError",
    "RetrievalMatch",
    "RetrievalResult",
    "UploadTooLargeError",
    "chunk_page_texts",
    "extract_page_texts",
    "normalize_document_id",
    "search_anchors",
]
