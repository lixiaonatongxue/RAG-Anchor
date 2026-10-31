"""Pure-function tests for the experimental semantic evidence selector.

These tests do not embed, read a database, or call a model. They assert exact
candidate IDs, offsets and final order rather than only counts.
"""

from __future__ import annotations

import copy
from uuid import UUID

from rag_anchor.domain.text import TEXT_EXTRACTION_VERSION
from rag_anchor.services.evidence_selection import (
    SELECTION_MAX_GAP,
    select_semantic_evidence,
    selection_pool_size,
)
from rag_anchor.services.semantic_candidates import build_candidate_id
from rag_anchor.services.semantic_service import SemanticMatch

DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000123")
OTHER_DOCUMENT_ID = UUID("00000000-0000-0000-0000-000000000456")


def _match(
    page_text: str,
    needle: str,
    score: float,
    *,
    page_index: int = 0,
    chunk_index: int = 0,
    document_id: UUID = DOCUMENT_ID,
) -> SemanticMatch:
    start = page_text.index(needle)
    return _span_match(
        page_text,
        start,
        start + len(needle),
        score,
        page_index=page_index,
        chunk_index=chunk_index,
        document_id=document_id,
    )


def _span_match(
    page_text: str,
    start: int,
    end: int,
    score: float,
    *,
    page_index: int = 0,
    chunk_index: int = 0,
    document_id: UUID = DOCUMENT_ID,
) -> SemanticMatch:
    return SemanticMatch(
        document_id=document_id,
        chunk_id=f"chunk-{page_index}-{chunk_index}",
        page_index=page_index,
        page_number=page_index + 1,
        chunk_index=chunk_index,
        start_offset=start,
        end_offset=end,
        source_text=page_text[start:end],
        score=score,
        extraction_version=TEXT_EXTRACTION_VERSION,
    )


def _ids(matches: list[SemanticMatch]) -> list[str]:
    return [
        build_candidate_id(
            document_id=match.document_id,
            page_index=match.page_index,
            start_offset=match.start_offset,
            end_offset=match.end_offset,
        )
        for match in matches
    ]


def _spans(matches: list[SemanticMatch]) -> list[tuple[int, int]]:
    return [(match.start_offset, match.end_offset) for match in matches]


def test_empty_and_single_candidate_pools() -> None:
    assert select_semantic_evidence([], limit=5) == []

    only = _match("Only sentence.", "Only sentence.", 0.8)
    assert select_semantic_evidence([only], limit=5) == [only]


def test_top_ranked_candidate_is_kept_even_when_input_is_unsorted() -> None:
    page = "Low sentence. High sentence."
    low = _match(page, "Low sentence.", 0.4)
    high = _match(page, "High sentence.", 0.9)

    selected = select_semantic_evidence([low, high], limit=1)

    assert selected == [high]
    assert _ids(selected) == _ids([high])


def test_nearest_same_page_neighbor_within_gap_is_selected() -> None:
    page = "Seed sentence. Neighbor sentence. Far sentence."
    seed = _match(page, "Seed sentence.", 0.90)
    neighbor = _match(page, "Neighbor sentence.", 0.85)
    far = _match(page, "Far sentence.", 0.80)

    selected = select_semantic_evidence([seed, far, neighbor], limit=3)

    assert _ids(selected) == _ids([seed, neighbor, far])
    assert _spans(selected) == [
        (0, 14),
        (15, 33),
        (34, 47),
    ]


def test_neighbor_below_the_seed_score_gap_is_not_replaced_by_a_farther_candidate() -> None:
    page = "Seed sentence. Weak neighbor. Strong farther sentence."
    seed = _match(page, "Seed sentence.", 0.90)
    weak = _match(page, "Weak neighbor.", 0.79)
    farther = _match(page, "Strong farther sentence.", 0.88)

    selected = select_semantic_evidence([seed, weak, farther], limit=2)

    assert SELECTION_MAX_GAP == 0.10
    assert _ids(selected) == _ids([seed, farther])
    assert weak not in selected


def test_cross_page_candidate_is_never_the_neighbor() -> None:
    page0 = "Seed sentence. Same page sentence."
    page1 = "Adjacent other page sentence."
    seed = _match(page0, "Seed sentence.", 0.90)
    same_page = _match(page0, "Same page sentence.", 0.80)
    other_page = _match(page1, "Adjacent other page sentence.", 0.89, page_index=1)

    selected = select_semantic_evidence([seed, other_page, same_page], limit=2)

    assert _ids(selected) == _ids([seed, same_page])
    assert _spans(selected) == [(0, 14), (15, 34)]


def test_overlapping_candidates_are_skipped_for_neighbor_and_fill() -> None:
    page = "Seed sentence. Overlapping tail sentence. Next sentence."
    seed = _span_match(page, 0, 14, 0.90)
    overlapping = _span_match(page, 10, 34, 0.89)
    next_sentence = _span_match(page, 42, 56, 0.80)

    selected = select_semantic_evidence(
        [seed, overlapping, next_sentence],
        limit=3,
    )

    assert _ids(selected) == _ids([seed, next_sentence])
    assert overlapping not in selected


def test_cross_chunk_same_page_candidate_can_be_the_neighbor() -> None:
    page = "Seed sentence. Neighbor in another chunk."
    seed = _match(page, "Seed sentence.", 0.90, chunk_index=0)
    neighbor = _match(page, "Neighbor in another chunk.", 0.86, chunk_index=1)

    selected = select_semantic_evidence([seed, neighbor], limit=2)

    assert _ids(selected) == _ids([seed, neighbor])
    assert selected[1].chunk_index == 1


def test_limit_and_pool_exhaustion_never_fabricate_evidence() -> None:
    page = "First sentence. Second sentence."
    first = _match(page, "First sentence.", 0.9)
    second = _match(page, "Second sentence.", 0.8)

    assert _ids(select_semantic_evidence([first, second], limit=20)) == _ids([first, second])
    assert _ids(select_semantic_evidence([first, second], limit=1)) == _ids([first])
    assert _ids(select_semantic_evidence([first, second], limit=2)) == _ids([first, second])

    assert selection_pool_size(1) == 3
    assert selection_pool_size(5) == 15
    assert selection_pool_size(6) == 18
    assert selection_pool_size(7) == 20
    assert selection_pool_size(20) == 20


def test_equal_scores_are_ordered_by_the_deterministic_rank_key() -> None:
    page = "A sentence. B sentence. C sentence. D sentence."
    matches = [
        _match(page, "C sentence.", 0.7),
        _match(page, "A sentence.", 0.7),
        _match(page, "D sentence.", 0.7),
        _match(page, "B sentence.", 0.7),
    ]

    first = select_semantic_evidence(matches, limit=4)
    second = select_semantic_evidence(list(reversed(matches)), limit=4)

    expected_order = [
        _match(page, "A sentence.", 0.7),
        _match(page, "B sentence.", 0.7),
        _match(page, "C sentence.", 0.7),
        _match(page, "D sentence.", 0.7),
    ]
    assert _ids(first) == _ids(expected_order)
    assert _ids(second) == _ids(expected_order)
    assert _spans(first) == sorted(_spans(first))


def test_selection_does_not_modify_the_input_or_its_objects() -> None:
    page = "Seed sentence. Neighbor sentence."
    seed = _match(page, "Seed sentence.", 0.9)
    neighbor = _match(page, "Neighbor sentence.", 0.85)
    matches = [neighbor, seed]
    before = copy.deepcopy(matches)

    select_semantic_evidence(matches, limit=2)

    assert matches == before
    assert [match.start_offset for match in matches] == [
        neighbor.start_offset,
        seed.start_offset,
    ]


def test_limit_one_returns_only_the_top_candidate() -> None:
    page = "Seed sentence. Neighbor sentence."
    seed = _match(page, "Seed sentence.", 0.9)
    neighbor = _match(page, "Neighbor sentence.", 0.89)

    assert _ids(select_semantic_evidence([seed, neighbor], limit=1)) == _ids([seed])


def test_duplicate_spans_are_kept_once_but_repeated_text_elsewhere_is_distinct() -> None:
    page = "Same sentence. Same sentence."
    first = _match(page, "Same sentence.", 0.9)
    duplicate = _span_match(
        page,
        first.start_offset,
        first.end_offset,
        0.9,
        chunk_index=1,
    )
    repeated = _span_match(
        page,
        page.rindex("Same sentence."),
        page.rindex("Same sentence.") + len("Same sentence."),
        0.8,
    )

    selected = select_semantic_evidence([first, duplicate, repeated], limit=3)

    assert _ids(selected) == _ids([first, repeated])
    assert selected[0].start_offset != selected[1].start_offset


def test_same_page_candidates_are_filled_without_a_page_cap() -> None:
    page = "One sentence. Two sentence. Three sentence. Four sentence."
    one = _match(page, "One sentence.", 0.9)
    two = _match(page, "Two sentence.", 0.8)
    three = _match(page, "Three sentence.", 0.7)
    four = _match(page, "Four sentence.", 0.6)

    selected = select_semantic_evidence([one, two, three, four], limit=4)

    assert _ids(selected) == _ids([one, two, three, four])
    assert len({match.page_index for match in selected}) == 1


def test_cross_document_candidates_are_not_neighbors() -> None:
    page0 = "Seed sentence. Same document sentence."
    page1 = "Foreign adjacent sentence."
    seed = _match(page0, "Seed sentence.", 0.9)
    same_document = _match(page0, "Same document sentence.", 0.8)
    foreign = _match(
        page1,
        "Foreign adjacent sentence.",
        0.89,
        document_id=OTHER_DOCUMENT_ID,
    )

    selected = select_semantic_evidence([seed, foreign, same_document], limit=2)

    assert _ids(selected) == _ids([seed, same_document])
    assert selected[0].document_id == DOCUMENT_ID
    assert all(match.document_id == DOCUMENT_ID for match in selected)
