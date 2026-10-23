"""Question answering endpoint with optional grounded generation."""

from __future__ import annotations

from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from rag_anchor.api.schemas import (
    EvidenceCitationResponse,
    QuestionRequest,
    QuestionResponse,
)
from rag_anchor.infra.llm import (
    LLMDisabledError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
)
from rag_anchor.services.document_service import DocumentNotFoundError
from rag_anchor.services.qa_service import (
    GeneratedAnswerError,
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
    """Return evidence-only citations or a grounded generated answer.

    ``evidence_only`` (the default) never calls a language model.
    ``grounded_generation`` sends the retrieved evidence to the configured
    provider and validates the returned citation IDs server-side. In both
    modes the citation fields come from the stored index, and callers must use
    ``POST /documents/{document_id}/evidence/locate`` before drawing geometry.

    Error contract:

    * 404 - the document has no metadata row.
    * 409 - the document has no usable chunk index.
    * 422 - the payload is invalid.
    * 502 - the model service failed or its output could not be verified.
    * 503 - grounded generation is disabled or not configured.
    * 504 - the model request timed out.
    """
    try:
        result = service.answer(
            document_id,
            payload.question,
            retrieval_query=payload.retrieval_query,
            match_case=payload.match_case,
            max_citations=payload.max_citations,
            answer_mode=payload.answer_mode,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except QuestionIndexUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # Defensive: the request model normally rejects these values first.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail="LLM request timed out") from exc
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail="LLM service request failed") from exc
    except (LLMResponseError, GeneratedAnswerError) as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM response could not be verified",
        ) from exc
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
