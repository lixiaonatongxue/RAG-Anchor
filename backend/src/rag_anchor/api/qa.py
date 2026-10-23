"""Evidence-only question answering endpoint."""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_anchor.api.schemas import (
    EvidenceCitationResponse,
    QuestionRequest,
    QuestionResponse,
)
from rag_anchor.services.document_service import DocumentNotFoundError
from rag_anchor.services.qa_service import (
    QuestionAnswerService,
    QuestionCitationScopeError,
    QuestionIndexUnavailableError,
)

router = APIRouter(prefix="/documents", tags=["qa"])


def get_question_service(request: Request) -> QuestionAnswerService:
    """Resolve the application-scoped question answering service."""
    return cast(QuestionAnswerService, request.app.state.question_service)


QuestionServiceDep = Annotated[QuestionAnswerService, Depends(get_question_service)]


@router.post("/{document_id}/qa", response_model=QuestionResponse)
def answer_question(
    document_id: UUID,
    payload: QuestionRequest,
    service: QuestionServiceDep,
) -> QuestionResponse:
    """Return literal retrieval citations for one question.

    This endpoint does not call a language model. ``answer`` is a fixed,
    explicitly labelled notice, and every citation is a retrieved page-text
    span scoped to ``document_id``. The caller must locate a citation through
    ``POST /documents/{document_id}/evidence/locate`` before drawing it.

    Error contract:

    * 404 — the document has no metadata row.
    * 409 — the document has no usable chunk index.
    * 422 — the payload is invalid.
    """
    try:
        result = service.answer(
            document_id,
            payload.question,
            retrieval_query=payload.retrieval_query,
            match_case=payload.match_case,
            max_citations=payload.max_citations,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except QuestionIndexUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # Defensive: the request model normally rejects these values first.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QuestionCitationScopeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Retrieval returned an invalid citation scope",
        ) from exc

    return QuestionResponse(
        document_id=result.document_id,
        question=result.question,
        retrieval_query=result.retrieval_query,
        match_case=result.match_case,
        answer_mode=result.answer_mode,
        status=result.status,
        answer=result.answer,
        total_matches=result.total_matches,
        citations=[
            EvidenceCitationResponse.model_validate(citation) for citation in result.citations
        ],
    )
