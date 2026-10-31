"""Deterministic selection of the final semantic evidence set (experimental).

``legacy`` is the production default: the semantic service returns exactly the
first ``max_citations`` ranked candidates and every one becomes a citation.

``neighbor_v1`` is an internal experiment. It retrieves a wider ranked pool and
then applies three deterministic rules:

1. keep the global top-ranked candidate;
2. extend it with at most one same-page, non-overlapping candidate whose score
   is within ``SELECTION_MAX_GAP`` of the top candidate;
3. fill the remaining slots by the original cosine ranking, skipping spans that
   overlap an already selected candidate.

The function is pure: it never embeds, reads storage, or rewrites an offset. It
only reorders and subsets the already verified ``SemanticMatch`` objects, so
every selected span remains an exact slice of the persisted page text.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_anchor.services.semantic_candidates import build_candidate_id
from rag_anchor.services.semantic_service import MAX_TOP_K, SemanticMatch

SELECTION_MODE_LEGACY = "legacy"
SELECTION_MODE_NEIGHBOR_V1 = "neighbor_v1"
SELECTION_MODES = frozenset({SELECTION_MODE_LEGACY, SELECTION_MODE_NEIGHBOR_V1})

SELECTION_POOL_MULTIPLIER = 3
"""The initial pool is three times the final citation cap."""

SELECTION_MAX_GAP = 0.10
"""Maximum cosine-score drop below the global top-1 candidate for the neighbor."""

MAX_NEIGHBOR_EXTENSIONS = 1
"""At most one adjacent candidate is added by the neighbor rule."""


def selection_pool_size(max_citations: int) -> int:
    """Return the ranked pool size used before neighbor selection.

    The semantic service caps ``top_k`` at ``MAX_TOP_K`` (20), so a large
    ``max_citations`` cannot request more than the service can rank. Because
    ``max_citations`` itself is at most 20, the pool is never smaller than the
    final cap.
    """
    return min(MAX_TOP_K, SELECTION_POOL_MULTIPLIER * max_citations)


def select_semantic_evidence(
    matches: Sequence[SemanticMatch],
    *,
    limit: int,
    max_gap: float = SELECTION_MAX_GAP,
) -> list[SemanticMatch]:
    """Select at most ``limit`` candidates from ``matches`` deterministically.

    Definitions used by ``neighbor_v1``:

    * **Rank** is ``(-score, page_index, chunk_index, start_offset,
      end_offset, candidate_id)``. The function sorts its input by this key
      and then keeps the top ``selection_pool_size(limit)`` entries, so the
      result does not depend on caller ordering.
    * **Overlap** means two candidates are on the same page and their
      half-open Python character spans intersect. Identical spans therefore
      count as overlap and are kept once.
    * **Neighbor** means a candidate on the same page as the top-1 seed, not
      overlapping the seed, closest by ``abs(start_offset - seed.start_offset)``.
      Cross-page candidates are never neighbors; candidates in another chunk
      of the same page are eligible. Ties are broken by higher score, then
      ``chunk_index``, ``start_offset``, ``end_offset`` and deterministic
      candidate id. Offsets, not reading-order assumptions, define distance,
      so abnormal layout order is handled without guessing.
    * The neighbor's score is compared with the **global top-1 score**, not the
      last selected candidate. If the nearest non-overlapping candidate is
      below ``top_score - max_gap``, no further candidate is considered for
      the single neighbor slot; its weaker score is not skipped in favour of
      a farther candidate.
    * Filling never adds a page cap or a page-first pass. Exact duplicate
      spans are kept once; identical text at a different span or page remains
      distinct evidence.

    The input sequence and its objects are never modified. If the pool is
    empty, ``[]`` is returned. If fewer candidates exist than ``limit`` (or
    overlaps remove some), every usable candidate is returned and nothing is
    fabricated.
    """
    ranked = sorted(matches, key=_rank_key)
    pool = ranked[: selection_pool_size(limit)]
    if not pool or limit <= 0:
        return []

    selected = [0]
    if limit > 1:
        neighbor = _nearest_same_page_neighbor(pool, selected)
        if neighbor is not None and pool[neighbor].score >= pool[0].score - max_gap:
            selected.append(neighbor)

    for index, candidate in enumerate(pool):
        if len(selected) >= limit:
            break
        if index in selected:
            continue
        if any(_overlaps(candidate, pool[selected_index]) for selected_index in selected):
            continue
        selected.append(index)

    return [pool[index] for index in sorted(selected)]


def _nearest_same_page_neighbor(
    pool: Sequence[SemanticMatch],
    selected: Sequence[int],
) -> int | None:
    anchor = pool[selected[0]]
    candidates = [
        index
        for index, candidate in enumerate(pool)
        if index != selected[0]
        and candidate.document_id == anchor.document_id
        and candidate.page_index == anchor.page_index
        and not any(_overlaps(candidate, pool[selected_index]) for selected_index in selected)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda index: _neighbor_key(pool, anchor, index))


def _neighbor_key(
    pool: Sequence[SemanticMatch],
    anchor: SemanticMatch,
    index: int,
) -> tuple[float, float, int, int, int, str]:
    candidate = pool[index]
    return (
        abs(candidate.start_offset - anchor.start_offset),
        -candidate.score,
        candidate.chunk_index,
        candidate.start_offset,
        candidate.end_offset,
        build_candidate_id(
            document_id=candidate.document_id,
            page_index=candidate.page_index,
            start_offset=candidate.start_offset,
            end_offset=candidate.end_offset,
        ),
    )


def _rank_key(match: SemanticMatch) -> tuple[float, int, int, int, int, str]:
    return (
        -match.score,
        match.page_index,
        match.chunk_index,
        match.start_offset,
        match.end_offset,
        build_candidate_id(
            document_id=match.document_id,
            page_index=match.page_index,
            start_offset=match.start_offset,
            end_offset=match.end_offset,
        ),
    )


def _overlaps(left: SemanticMatch, right: SemanticMatch) -> bool:
    return (
        left.document_id == right.document_id
        and left.page_index == right.page_index
        and left.start_offset < right.end_offset
        and right.start_offset < left.end_offset
    )


__all__ = [
    "MAX_NEIGHBOR_EXTENSIONS",
    "SELECTION_MAX_GAP",
    "SELECTION_MODES",
    "SELECTION_MODE_LEGACY",
    "SELECTION_MODE_NEIGHBOR_V1",
    "SELECTION_POOL_MULTIPLIER",
    "select_semantic_evidence",
    "selection_pool_size",
]
