"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from rag_anchor import __version__
from rag_anchor.api.documents import router as documents_router
from rag_anchor.api.evidence import router as evidence_router
from rag_anchor.api.qa import router as qa_router
from rag_anchor.api.retrieval import router as retrieval_router
from rag_anchor.api.semantic import router as semantic_router
from rag_anchor.config import Settings
from rag_anchor.infra.embedding import EmbeddingProvider, build_embedding_provider
from rag_anchor.infra.llm import LLMProvider, build_llm_provider
from rag_anchor.services.document_service import DocumentService
from rag_anchor.services.evidence_service import EvidenceLocationService
from rag_anchor.services.index_service import DocumentIndexService
from rag_anchor.services.qa_service import QuestionAnswerService
from rag_anchor.services.retrieval_service import ChunkRetrievalService
from rag_anchor.services.semantic_service import SemanticIndexService

API_PREFIX = "/api/v1"


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    version: str


def create_app(
    settings: Settings | None = None,
    *,
    llm_provider: LLMProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Filesystem and database setup happens in the lifespan handler, not at
    import time, so importing the module has no side effects. The optional
    ``llm_provider`` and ``embedding_provider`` are injection points for
    offline tests; production uses providers built from environment settings,
    which are disabled by default and never probed during construction.
    """
    resolved_settings = settings or Settings.from_env()
    document_service = DocumentService(resolved_settings)
    index_service = DocumentIndexService(document_service)
    retrieval_service = ChunkRetrievalService(document_service)
    evidence_service = EvidenceLocationService(document_service)
    resolved_llm_provider = (
        llm_provider if llm_provider is not None else build_llm_provider(resolved_settings.llm)
    )
    resolved_embedding_provider = (
        embedding_provider
        if embedding_provider is not None
        else build_embedding_provider(resolved_settings.embedding)
    )
    semantic_service = SemanticIndexService(
        document_service,
        resolved_embedding_provider,
        settings=resolved_settings.embedding,
    )
    question_service = QuestionAnswerService(
        document_service,
        retrieval_service,
        resolved_llm_provider,
        semantic_service=semantic_service,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        document_service.initialize()
        yield

    app = FastAPI(
        title="RAG-AnchorLab API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.document_service = document_service
    app.state.index_service = index_service
    app.state.retrieval_service = retrieval_service
    app.state.evidence_service = evidence_service
    app.state.semantic_service = semantic_service
    app.state.question_service = question_service
    app.include_router(documents_router, prefix=API_PREFIX)
    app.include_router(retrieval_router, prefix=API_PREFIX)
    app.include_router(evidence_router, prefix=API_PREFIX)
    app.include_router(semantic_router, prefix=API_PREFIX)
    app.include_router(qa_router, prefix=API_PREFIX)

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app


app = create_app()
