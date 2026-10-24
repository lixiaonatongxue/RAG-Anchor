"""Deterministic, exactly traceable candidate spans for semantic retrieval.

An embedding can only rank text; it must never guess PDF offsets. This module
therefore builds every semantic candidate from the *persisted* chunk text and
records an exact slice of the stored page text::

    page_text[start_offset:end_offset] == text

and keeps the span inside its source chunk. A candidate that cannot satisfy
these invariants is rejected instead of being repaired approximately.

Candidates are sentence-like ranges split on terminal punctuation, then cut
into fixed-size windows when a sentence is implausibly long. Line breaks are
kept inside a candidate so a sentence wrapped by the PDF layout is not split
into two unrelated vectors. The resulting offsets remain Python ``str``
offsets, never PDF coordinates or PDFium character indices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from rag_anchor.domain.text import PageText, TextChunk

SEMANTIC_CANDIDATE_BUILDER_VERSION = 1
"""Version of the deterministic candidate split/offset rules.

Persisted with every derived index. Changing how sentences are split or
windows are cut changes which vectors exist, so an index built by another
builder version must be rebuilt instead of queried.
"""

MAX_CANDIDATE_CHARS = 500
"""Sentences longer than this are cut into fixed-size windows."""

WINDOW_CHARS = 400
"""Window length used for long sentences."""

MIN_CANDIDATE_CHARS = 2
"""Candidates shorter than this (after trimming) are not indexed."""

_CJK_TERMINATORS = frozenset("。！？")  # noqa: RUF001 - intentional CJK punctuation
_LATIN_TERMINATORS = frozenset(".!?")
_CLOSING_WRAPPERS = frozenset("”’\"')]}）】》」』〉")  # noqa: RUF001

_CANDIDATE_ID_PREFIX = "sem_"
_CANDIDATE_ID_HEX_LENGTH = 32


@dataclass(frozen=True, slots=True)
class SemanticCandidate:
    """One exact page-text span eligible for semantic ranking."""

    document_id: UUID
    chunk_id: str
    page_index: int
    page_number: int
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str
    extraction_version: str

    @property
    def candidate_id(self) -> str:
        """Deterministic identifier derived only from the exact page span."""
        return build_candidate_id(
            document_id=self.document_id,
            page_index=self.page_index,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )


def build_semantic_candidates(
    pages: Sequence[PageText],
    chunks: Sequence[TextChunk],
) -> list[SemanticCandidate]:
    """Return deterministic candidates grounded in stored page text.

    Every chunk is verified against its stored page before any span is
    produced. Candidates are ordered by ``(page_index, start_offset,
    chunk_index)``; identical absolute spans found in overlapping chunks are
    emitted once, attributed to the lowest ``chunk_index``.
    """
    pages_by_index = {page.page_index: page for page in pages}
    if len(pages_by_index) != len(pages):
        raise ValueError("pages must not contain duplicate page_index values")

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: (chunk.page_index, chunk.chunk_index, chunk.start_offset),
    )

    candidates: list[SemanticCandidate] = []
    seen_spans: set[tuple[str, int, int, int]] = set()

    for chunk in ordered_chunks:
        page = pages_by_index.get(chunk.page_index)
        if page is None:
            raise ValueError("chunk references a page that was not stored")
        chunk.verify_source(page)

        text = chunk.text
        for local_start, local_end in _candidate_ranges(text):
            local_start, local_end = _trim_space(text, local_start, local_end)
            if local_end - local_start < MIN_CANDIDATE_CHARS:
                continue

            start_offset = chunk.start_offset + local_start
            end_offset = chunk.start_offset + local_end
            source_text = text[local_start:local_end]
            if page.text[start_offset:end_offset] != source_text:
                raise ValueError("semantic candidate is not an exact page-text slice")
            if start_offset < chunk.start_offset or end_offset > chunk.end_offset:
                raise ValueError("semantic candidate is outside its source chunk")

            span_key = (chunk.document_id, chunk.page_index, start_offset, end_offset)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            candidates.append(
                SemanticCandidate(
                    document_id=UUID(chunk.document_id),
                    chunk_id=chunk.chunk_id,
                    page_index=chunk.page_index,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    text=source_text,
                    extraction_version=chunk.extraction_version,
                )
            )

    candidates.sort(key=lambda item: (item.page_index, item.start_offset, item.chunk_index))
    return candidates


def build_candidate_id(
    *,
    document_id: UUID,
    page_index: int,
    start_offset: int,
    end_offset: int,
) -> str:
    """Return a deterministic candidate identifier for one exact span."""
    payload = json.dumps(
        [str(document_id), page_index, start_offset, end_offset],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{_CANDIDATE_ID_PREFIX}{digest[:_CANDIDATE_ID_HEX_LENGTH]}"


def _candidate_ranges(text: str) -> list[tuple[int, int]]:
    """Split ``text`` on terminal punctuation, then window long sentences."""
    ranges: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        if _is_terminator(text, index):
            end = index + 1
            while end < length and text[end] in _CLOSING_WRAPPERS:
                end += 1
            ranges.extend(_window(text, start, end))
            start = end
            index = end
            continue
        index += 1
    if start < length:
        ranges.extend(_window(text, start, length))
    return ranges


def _window(text: str, start: int, end: int) -> list[tuple[int, int]]:
    if end - start <= MAX_CANDIDATE_CHARS:
        return [(start, end)]

    windows: list[tuple[int, int]] = []
    position = start
    half_window = WINDOW_CHARS // 2
    while position < end:
        tentative = min(position + WINDOW_CHARS, end)
        if tentative < end:
            whitespace = text.rfind(" ", position + half_window, tentative)
            if whitespace >= position + half_window:
                tentative = whitespace + 1
        windows.append((position, tentative))
        position = tentative
    return windows


def _trim_space(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _is_terminator(text: str, index: int) -> bool:
    char = text[index]
    if char in _CJK_TERMINATORS:
        return True
    if char not in _LATIN_TERMINATORS:
        return False

    following = index + 1
    while following < len(text) and text[following] in _CLOSING_WRAPPERS:
        following += 1
    if following >= len(text):
        return True
    return text[following].isspace()


__all__ = [
    "MAX_CANDIDATE_CHARS",
    "MIN_CANDIDATE_CHARS",
    "SEMANTIC_CANDIDATE_BUILDER_VERSION",
    "WINDOW_CHARS",
    "SemanticCandidate",
    "build_candidate_id",
    "build_semantic_candidates",
]
