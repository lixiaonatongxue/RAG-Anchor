"""Shared fixtures for API-level tests.

PDFs are generated at runtime; no binary document is stored in the repository.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from rag_anchor.config import Settings
from rag_anchor.infra.embedding import EmbeddingProvider
from rag_anchor.infra.llm import LLMProvider
from rag_anchor.main import create_app


def make_pdf_bytes(page_count: int = 1) -> bytes:
    """Return a small valid PDF with ``page_count`` pages."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for index in range(page_count):
        pdf.drawString(72, 720, f"Page {index + 1}")
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class ApiContext:
    """An application instance plus its live test client."""

    app: FastAPI
    client: TestClient
    settings: Settings


@pytest.fixture
def pdf_builder() -> Callable[..., bytes]:
    """Return the runtime PDF generator."""
    return make_pdf_bytes


@pytest.fixture
def pdf_bytes() -> bytes:
    """A one-page valid PDF."""
    return make_pdf_bytes()


@pytest.fixture
def api(tmp_path: Path) -> Iterator[ApiContext]:
    """Start an application with an isolated data directory."""
    settings = Settings(data_dir=tmp_path / "data")
    application = create_app(settings)
    with TestClient(application) as client:
        yield ApiContext(app=application, client=client, settings=settings)


@pytest.fixture
def api_factory(tmp_path: Path) -> Iterator[Callable[..., ApiContext]]:
    """Build multiple live applications sharing ``tmp_path``.

    The context manager stack keeps lifespan startup/shutdown balanced even
    when a test creates more than one application.
    """
    stack = ExitStack()

    def build(
        *,
        data_dir: Path | None = None,
        max_upload_bytes: int | None = None,
        llm_provider: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_selection_mode: str | None = None,
    ) -> ApiContext:
        kwargs: dict[str, object] = {"data_dir": data_dir or tmp_path / "data"}
        if max_upload_bytes is not None:
            kwargs["max_upload_bytes"] = max_upload_bytes
        settings = Settings(**kwargs)  # type: ignore[arg-type]
        app_kwargs: dict[str, object] = {
            "llm_provider": llm_provider,
            "embedding_provider": embedding_provider,
        }
        if semantic_selection_mode is not None:
            app_kwargs["semantic_selection_mode"] = semantic_selection_mode
        application = create_app(settings, **app_kwargs)  # type: ignore[arg-type]
        client = stack.enter_context(TestClient(application))
        return ApiContext(app=application, client=client, settings=settings)

    try:
        yield build
    finally:
        stack.close()
