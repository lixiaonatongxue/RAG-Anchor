"""Exact, verifiable support spans for retrieved citation hits.

A retrieval hit is a literal keyword occurrence. Highlighting only that keyword
usually shows too little of the source sentence, so citations carry a support
span that is the complete sentence containing the hit when that sentence can be
determined reliably.

Every returned span is an exact slice of the stored page text::

    page_text[support_start:support_end] == support_text

The span is always contained in the retrieved chunk, so the existing
``evidence/locate`` contract (which requires a span inside the chunk when a
``chunk_id`` is supplied) still applies unchanged.

Sentence detection is a deterministic punctuation heuristic, **not** a proof
that the sentence supports a generated conclusion. When the heuristic cannot
produce a reliable sentence -- the hit itself crosses a sentence boundary or a
line break, a boundary is missing, the sentence is cut by the chunk boundary or
the sentence is implausibly long -- the original keyword hit is returned and
explicitly marked as a retrieval-only span.
"""

from __future__ import annotations

from dataclasses import dataclass

SUPPORT_KIND_SENTENCE = "sentence"
SUPPORT_KIND_RETRIEVAL_MATCH = "retrieval_match"
SUPPORT_KINDS = frozenset({SUPPORT_KIND_SENTENCE, SUPPORT_KIND_RETRIEVAL_MATCH})

MAX_SUPPORT_TEXT_LENGTH = 500
"""Longer candidate sentences fall back to the retrieval-only hit span."""

_CJK_TERMINATORS = frozenset("。！？")  # noqa: RUF001 - intentional CJK punctuation
_LATIN_TERMINATORS = frozenset(".!?")
_LINE_BREAKS = frozenset("\r\n\u2028\u2029")
_CLOSING_WRAPPERS = frozenset("”’\"')]}）】》」』〉")  # noqa: RUF001


@dataclass(frozen=True, slots=True)
class SupportSpan:
    """One exact page-text span offered as citation support."""

    start: int
    end: int
    text: str
    kind: str


def build_support_span(
    page_text: str,
    *,
    chunk_start: int,
    chunk_end: int,
    match_start: int,
    match_end: int,
) -> SupportSpan:
    """Return the sentence around ``[match_start, match_end)`` or the hit itself.

    All offsets are Python ``str`` indices into ``page_text``. The hit and the
    chunk must lie inside the page text; the hit must lie inside the chunk.
    """
    _validate_bounds(page_text, chunk_start, chunk_end, match_start, match_end)

    hit_text = page_text[match_start:match_end]
    fallback = SupportSpan(
        start=match_start,
        end=match_end,
        text=hit_text,
        kind=SUPPORT_KIND_RETRIEVAL_MATCH,
    )
    if not hit_text:
        return fallback
    if _contains_line_break(hit_text) or _contains_terminator(page_text, match_start, match_end):
        # The hit already spans several lines or sentences, so no single
        # sentence can be selected without guessing.
        return fallback

    start = _sentence_start(page_text, chunk_start, match_start)
    if start is None:
        return fallback
    end = _sentence_end(page_text, chunk_end, match_end)
    if end is None:
        return fallback

    while start < match_start and page_text[start].isspace():
        start += 1
    while end > match_end and page_text[end - 1].isspace():
        end -= 1

    if start > match_start or end < match_end or end <= start:
        return fallback

    support_text = page_text[start:end]
    if (
        len(support_text) > MAX_SUPPORT_TEXT_LENGTH
        or _contains_line_break(support_text)
        or not support_text.strip()
    ):
        return fallback

    return SupportSpan(
        start=start,
        end=end,
        text=support_text,
        kind=SUPPORT_KIND_SENTENCE,
    )


def _validate_bounds(
    page_text: object,
    chunk_start: object,
    chunk_end: object,
    match_start: object,
    match_end: object,
) -> None:
    if not isinstance(page_text, str):
        raise TypeError("page_text must be a string")
    for name, value in (
        ("chunk_start", chunk_start),
        ("chunk_end", chunk_end),
        ("match_start", match_start),
        ("match_end", match_end),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 <= chunk_start <= match_start < match_end <= chunk_end <= len(page_text):
        raise ValueError(
            "span must satisfy 0 <= chunk_start <= match_start < match_end "
            "<= chunk_end <= len(page_text)"
        )


def _contains_line_break(value: str) -> bool:
    return any(char in _LINE_BREAKS for char in value)


def _contains_terminator(page_text: str, start: int, end: int) -> bool:
    return any(_is_terminator(page_text, index) for index in range(start, end))


def _sentence_start(page_text: str, chunk_start: int, match_start: int) -> int | None:
    index = match_start - 1
    while index >= chunk_start:
        if _is_terminator(page_text, index):
            start = index + 1
            while start < match_start and (
                page_text[start].isspace() or page_text[start] in _CLOSING_WRAPPERS
            ):
                start += 1
            return start
        index -= 1

    if chunk_start == 0:
        start = 0
        while start < match_start and page_text[start].isspace():
            start += 1
        return start
    return None


def _sentence_end(page_text: str, chunk_end: int, match_end: int) -> int | None:
    index = match_end
    while index < chunk_end:
        if _is_terminator(page_text, index):
            end = index + 1
            while end < chunk_end and page_text[end] in _CLOSING_WRAPPERS:
                end += 1
            return end
        index += 1

    # A chunk that happens to end at the page end is still not proof that the
    # text after the hit forms a complete sentence; require terminal
    # punctuation inside the chunk instead of guessing.
    return None


def _is_terminator(page_text: str, index: int) -> bool:
    char = page_text[index]
    if char in _CJK_TERMINATORS:
        return True
    if char not in _LATIN_TERMINATORS:
        return False

    # A Latin period inside a number or abbreviation ("3.14", "U.S") is not a
    # sentence boundary unless it is followed by whitespace or the end of the
    # text. Closing quotes and brackets may sit between the period and the
    # whitespace.
    following = index + 1
    while following < len(page_text) and page_text[following] in _CLOSING_WRAPPERS:
        following += 1
    if following >= len(page_text):
        return True
    return page_text[following].isspace()


__all__ = [
    "MAX_SUPPORT_TEXT_LENGTH",
    "SUPPORT_KINDS",
    "SUPPORT_KIND_RETRIEVAL_MATCH",
    "SUPPORT_KIND_SENTENCE",
    "SupportSpan",
    "build_support_span",
]
