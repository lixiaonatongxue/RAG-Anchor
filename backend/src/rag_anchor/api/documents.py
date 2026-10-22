"""Document upload and management endpoints."""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse

from rag_anchor.api.schemas import (
    AnchorHitResponse,
    AnchorSearchRequest,
    AnchorSearchResponse,
    DocumentIndexResponse,
    DocumentListResponse,
    DocumentResponse,
)
from rag_anchor.services.anchor_service import PdfAnchorError, search_anchors
from rag_anchor.services.document_service import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentService,
    InvalidUploadError,
    UploadTooLargeError,
)
from rag_anchor.services.index_service import (
    DocumentIndexPersistenceError,
    DocumentIndexService,
    DocumentIndexSourceError,
)
from rag_anchor.services.text_extraction_service import PdfTextExtractionError

router = APIRouter(prefix="/documents", tags=["documents"])

_SAFE_DOWNLOAD_FILENAME = "document.pdf"
"""Fixed Content-Disposition filename; the client's original name is never used."""


def get_document_service(request: Request) -> DocumentService:
    """Resolve the application-scoped document service."""
    return cast(DocumentService, request.app.state.document_service)


ServiceDep = Annotated[DocumentService, Depends(get_document_service)]


def get_index_service(request: Request) -> DocumentIndexService:
    """Resolve the application-scoped document index service."""
    return cast(DocumentIndexService, request.app.state.index_service)


IndexServiceDep = Annotated[DocumentIndexService, Depends(get_index_service)]


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF file to store")],
    service: ServiceDep,
) -> DocumentResponse:
    """Upload and validate one PDF."""
    try:
        record = await service.create_document(file)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentPersistenceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return DocumentResponse.model_validate(record)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    service: ServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    """List stored documents, newest first."""
    records, total = service.list_documents(limit=limit, offset=offset)
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(record) for record in records],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: UUID, service: ServiceDep) -> DocumentResponse:
    """Return metadata for one document."""
    record = service.get_document(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse.model_validate(record)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: UUID, service: ServiceDep) -> None:
    """Delete one document and its stored PDF."""
    try:
        service.delete_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@router.post("/{document_id}/anchors/search", response_model=AnchorSearchResponse)
def search_document_anchors(
    document_id: UUID,
    payload: AnchorSearchRequest,
    service: ServiceDep,
) -> AnchorSearchResponse:
    """Return every occurrence of the query in a stored PDF.

    This route is intentionally synchronous so FastAPI runs the blocking
    PDFium work in its worker thread pool instead of the event loop.
    """
    try:
        pdf_path = service.get_document_path(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc

    try:
        hits = search_anchors(pdf_path, payload.query, match_case=payload.match_case)
    except OSError as exc:
        # FileNotFoundError / IsADirectoryError / permission errors. The
        # original message contains the absolute path and must not leak.
        raise HTTPException(
            status_code=500,
            detail="Stored PDF is missing or unreadable",
        ) from exc
    except PdfAnchorError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored PDF could not be searched",
        ) from exc
    except ValueError as exc:
        # Defensive: the request model normally rejects an empty query first.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    results = [AnchorHitResponse.model_validate(hit) for hit in hits]
    return AnchorSearchResponse(
        document_id=document_id,
        query=payload.query,
        match_case=payload.match_case,
        total=len(results),
        results=results,
    )


@router.post("/{document_id}/index", response_model=DocumentIndexResponse)
def index_document(
    document_id: UUID,
    service: IndexServiceDep,
) -> DocumentIndexResponse:
    """Build or rebuild the traceable text index of one stored PDF.

    The whole pipeline (PDF hashing, text extraction, chunking, source
    verification and the atomic index replacement) is delegated to
    :meth:`DocumentIndexService.index_document`. This route only maps domain
    failures to HTTP status codes and never writes index rows itself.

    The route is synchronous so FastAPI runs the blocking PDFium and SQLite
    work in its worker thread pool instead of the event loop.

    Repeating the call rebuilds the index with the service defaults; an
    unchanged PDF ends up with the same rows, not with duplicated chunks.
    A failed rebuild leaves the previous index untouched because the
    repository replaces the index in one transaction.
    """
    try:
        result = service.index_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except DocumentIndexSourceError as exc:
        # The original message contains the absolute stored path.
        raise HTTPException(
            status_code=500,
            detail="Stored PDF is missing or unreadable",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored PDF is missing or unreadable",
        ) from exc
    except PdfTextExtractionError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored PDF could not be indexed",
        ) from exc
    except DocumentIndexPersistenceError as exc:
        raise HTTPException(
            status_code=500,
            detail="Document index could not be stored",
        ) from exc
    except ValueError as exc:
        # Defensive: the first version exposes no chunking parameters.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DocumentIndexResponse.model_validate(result)


@router.get(
    "/{document_id}/file",
    response_class=FileResponse,
    responses={
        200: {"content": {"application/pdf": {}}},
        404: {"description": "Document metadata not found"},
        500: {"description": "Stored PDF is missing or unreadable"},
    },
)
def get_document_file(document_id: UUID, service: ServiceDep) -> FileResponse:
    """Return the stored PDF bytes for inline browser display.

    The path is always derived from the UUID by the service, never from a
    client-supplied filename. Missing metadata is a 404; a missing or
    unreadable on-disk file is a generic 500 that never reveals the path.
    """
    try:
        pdf_path = service.get_document_path(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc

    if not pdf_path.is_file():
        raise HTTPException(
            status_code=500,
            detail="Stored PDF is missing or unreadable",
        )

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=_SAFE_DOWNLOAD_FILENAME,
        content_disposition_type="inline",
    )
