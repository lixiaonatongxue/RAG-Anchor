"""FastAPI application factory and ASGI entry point."""

from fastapi import FastAPI
from pydantic import BaseModel

from rag_anchor import __version__

API_PREFIX = "/api/v1"


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    version: str


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(title="RAG-AnchorLab API", version=__version__)

    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app


app = create_app()
