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
from rag_anchor.infra.embedding import (
    EmbeddingDisabledError,
    EmbeddingResponseError,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
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
from rag_anchor.services.semantic_service import (
    SemanticBaseIndexUnavailableError,
    SemanticIndexEmptyError,
    SemanticIndexStaleError,
    SemanticIndexUnavailableError,
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
    provider and validates the returned citation IDs server-side.
    ``retrieval_mode`` defaults to literal keyword retrieval; ``semantic``
    requires a separately built derived index and never falls back to literal
    matching. In every mode the citation fields come from the stored index,
    and callers must use ``POST /documents/{document_id}/evidence/locate``
    before drawing geometry.

    Error contract:

    * 404 - the document has no metadata row.
    * 409 - the document has no usable chunk index, or the semantic index is
      missing, empty or stale.
    * 422 - the payload is invalid.
    * 502 - the model or embedding service failed or returned unverifiable data.
    * 503 - grounded generation or semantic retrieval is disabled/unconfigured.
    * 504 - the model or embedding request timed out.
    """
    try:
        result = service.answer(
            document_id,
            payload.question,
            retrieval_query=payload.retrieval_query,
            retrieval_mode=payload.retrieval_mode,
            match_case=payload.match_case,
            max_citations=payload.max_citations,
            answer_mode=payload.answer_mode,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc
    except QuestionIndexUnavailableError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (
        SemanticBaseIndexUnavailableError,
        SemanticIndexUnavailableError,
        SemanticIndexEmptyError,
        SemanticIndexStaleError,
    ) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        # Defensive: the request model normally rejects these values first.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail="LLM request timed out") from exc
    except EmbeddingTimeoutError as exc:
        raise HTTPException(status_code=504, detail="Embedding request timed out") from exc
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail="LLM service request failed") from exc
    except (LLMResponseError, GeneratedAnswerError) as exc:
        raise HTTPException(
            status_code=502,
            detail="LLM response could not be verified",
        ) from exc
    except (EmbeddingServiceError, EmbeddingResponseError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Embedding service request failed or returned invalid vectors",
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
        retrieval_mode=result.retrieval_mode,
        match_case=result.match_case,
        answer_mode=result.answer_mode,
        status=result.status,
        answer=result.answer,
        total_matches=result.total_matches,
        citations=[
            EvidenceCitationResponse.model_validate(citation) for citation in result.citations
        ],
    )
