"""Document upload and management endpoints."""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile

from rag_anchor.api.schemas import DocumentListResponse, DocumentResponse
from rag_anchor.services.document_service import (
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentService,
    InvalidUploadError,
    UploadTooLargeError,
)

router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service(request: Request) -> DocumentService:
    """Resolve the application-scoped document service."""
    return cast(DocumentService, request.app.state.document_service)


ServiceDep = Annotated[DocumentService, Depends(get_document_service)]


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
