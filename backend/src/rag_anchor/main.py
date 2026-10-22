"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from rag_anchor import __version__
from rag_anchor.api.documents import router as documents_router
from rag_anchor.api.retrieval import router as retrieval_router
from rag_anchor.config import Settings
from rag_anchor.services.document_service import DocumentService
from rag_anchor.services.index_service import DocumentIndexService
from rag_anchor.services.retrieval_service import ChunkRetrievalService

API_PREFIX = "/api/v1"


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application.

    Filesystem and database setup happens in the lifespan handler, not at
    import time, so importing the module has no side effects.
    """
    resolved_settings = settings or Settings.from_env()
    document_service = DocumentService(resolved_settings)
    index_service = DocumentIndexService(document_service)
    retrieval_service = ChunkRetrievalService(document_service)

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
    app.include_router(documents_router, prefix=API_PREFIX)
    app.include_router(retrieval_router, prefix=API_PREFIX)

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app


app = create_app()
