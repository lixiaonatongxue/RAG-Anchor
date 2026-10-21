"""Traceable retrieval over persisted document chunks."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_anchor.api.schemas import (
    ChunkRetrievalRequest,
    ChunkRetrievalResponse,
    RetrievalHitResponse,
)
from rag_anchor.services.document_service import DocumentNotFoundError
from rag_anchor.services.retrieval_service import ChunkRetrievalService

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def get_retrieval_service(request: Request) -> ChunkRetrievalService:
    """Resolve the application-scoped chunk retrieval service."""
    return cast(ChunkRetrievalService, request.app.state.retrieval_service)


RetrievalDep = Annotated[ChunkRetrievalService, Depends(get_retrieval_service)]


@router.post("/search", response_model=ChunkRetrievalResponse)
def search_chunks(
    payload: ChunkRetrievalRequest,
    service: RetrievalDep,
) -> ChunkRetrievalResponse:
    """Return traceable chunk hits across all documents or one document.

    The route is synchronous so FastAPI runs the blocking SQLite read in its
    worker thread pool instead of blocking the event loop.
    """
    try:
        result = service.search_chunks(
            payload.query,
            document_id=payload.document_id,
            match_case=payload.match_case,
            limit=payload.limit,
            offset=payload.offset,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except ValueError as exc:
        # Defensive: the request model normally rejects these values first.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ChunkRetrievalResponse(
        query=result.query,
        match_case=result.match_case,
        document_id=result.document_id,
        total=result.total,
        limit=result.limit,
        offset=result.offset,
        results=[RetrievalHitResponse.model_validate(match) for match in result.matches],
    )
