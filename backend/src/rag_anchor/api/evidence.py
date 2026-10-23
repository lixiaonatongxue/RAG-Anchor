"""Exact evidence location endpoint over the persisted index."""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pypdfium2 import PdfiumError

from rag_anchor.api.schemas import (
    EvidenceLocateRequest,
    EvidenceLocateResponse,
    LocatedEvidenceResponse,
)
from rag_anchor.services.evidence_service import (
    EvidenceDocumentNotFoundError,
    EvidenceIndexMismatchError,
    EvidenceIndexUnavailableError,
    EvidenceLocationService,
    EvidenceMappingError,
    EvidenceSourceError,
    EvidenceSpan,
    EvidenceValidationError,
)

router = APIRouter(prefix="/documents", tags=["evidence"])


def get_evidence_service(request: Request) -> EvidenceLocationService:
    """Resolve the application-scoped evidence location service."""
    return cast(EvidenceLocationService, request.app.state.evidence_service)


EvidenceServiceDep = Annotated[EvidenceLocationService, Depends(get_evidence_service)]


@router.post("/{document_id}/evidence/locate", response_model=EvidenceLocateResponse)
def locate_evidence(
    document_id: UUID,
    payload: EvidenceLocateRequest,
    service: EvidenceServiceDep,
) -> EvidenceLocateResponse:
    """Locate exact page-text spans in the stored rendered PDF.

    Each item identifies one occurrence by ``page_index`` plus Python string
    offsets into the stored page text; ``hit_text`` is only verified, never
    searched. The route is synchronous so FastAPI runs the blocking PDFium
    and SQLite work in its worker thread pool instead of the event loop.

    Error contract (specific to this endpoint):

    * 404 — the document has no metadata row.
    * 422 — a request item is invalid against the stored page text or its
      optional ``chunk_id`` does not match the requested span.
    * 409 — the document is not indexed, the stored index no longer matches
      the current PDF, or a valid span cannot be mapped to an exact PDFium
      character range with real geometry.
    * 500 — the stored PDF is missing, unreadable or unusable.

    The whole batch fails if any item fails; partial results are never
    returned.
    """
    spans = [
        EvidenceSpan(
            page_index=item.page_index,
            match_start=item.match_start,
            match_end=item.match_end,
            hit_text=item.hit_text,
            chunk_id=item.chunk_id,
        )
        for item in payload.items
    ]

    try:
        located = service.locate(document_id, spans)
    except EvidenceDocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except EvidenceValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (
        EvidenceIndexUnavailableError,
        EvidenceIndexMismatchError,
        EvidenceMappingError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EvidenceSourceError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored PDF is missing or unreadable",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored PDF is missing or unreadable",
        ) from exc
    except PdfiumError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored PDF could not be read",
        ) from exc

    results = [LocatedEvidenceResponse.model_validate(item) for item in located]
    return EvidenceLocateResponse(
        document_id=document_id,
        total=len(results),
        results=results,
    )
