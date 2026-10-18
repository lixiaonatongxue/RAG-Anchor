"""Pydantic models exposed by the document API."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Public metadata for one stored PDF.

    The server-side absolute path is intentionally absent.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    filename: str
    size_bytes: int = Field(ge=0)
    page_count: int = Field(ge=1)
    created_at: datetime


class DocumentListResponse(BaseModel):
    """One page of document metadata."""

    items: list[DocumentResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
