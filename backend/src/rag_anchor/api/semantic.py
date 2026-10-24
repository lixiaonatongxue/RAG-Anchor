"""Explicit semantic index build endpoint.

The derived vector index is never built implicitly. This route is synchronous
so FastAPI runs the blocking embedding requests and SQLite transaction in its
worker thread pool. Rebuilding replaces the previous vectors in one
transaction; a failed run leaves the previous derived index untouched.
"""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_anchor.api.schemas import SemanticIndexResponse
from rag_anchor.infra.embedding import (
    EmbeddingDisabledError,
    EmbeddingResponseError,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
)
from rag_anchor.services.document_service import DocumentNotFoundError
from rag_anchor.services.semantic_service import (
    SemanticBaseIndexUnavailableError,
    SemanticIndexEmptyError,
    SemanticIndexPersistenceError,
    SemanticIndexService,
    SemanticIndexStaleError,
)

router = APIRouter(prefix="/documents", tags=["semantic"])


def get_semantic_service(request: Request) -> SemanticIndexService:
    """Resolve the application-scoped semantic index service."""
    return cast(SemanticIndexService, request.app.state.semantic_service)


SemanticServiceDep = Annotated[SemanticIndexService, Depends(get_semantic_service)]


@router.post("/{document_id}/semantic-index", response_model=SemanticIndexResponse)
def build_semantic_index(
    document_id: UUID,
    service: SemanticServiceDep,
) -> SemanticIndexResponse:
    """Embed and persist every candidate of one already text-indexed document.

    Error contract:

    * 404 - the document has no metadata row.
    * 409 - the text index is missing/stale or the document has no candidates.
    * 422 - the stored text cannot produce valid candidates.
    * 502 - the embedding service failed or returned invalid vectors.
    * 503 - semantic retrieval is disabled or not configured.
    * 504 - the embedding request timed out.
    * 500 - the derived index could not be stored.
    """
    try:
        result = service.build_index(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except (
        SemanticBaseIndexUnavailableError,
        SemanticIndexEmptyError,
        SemanticIndexStaleError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmbeddingDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Embedding request timed out") from exc
    except (EmbeddingServiceError, EmbeddingResponseError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Embedding service request failed or returned invalid vectors",
        ) from exc
    except SemanticIndexPersistenceError as exc:
        raise HTTPException(
            status_code=500,
            detail="Semantic index could not be stored",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SemanticIndexResponse.model_validate(result)
