"""One-to-one prediction/truth matching and failure classification."""

from __future__ import annotations

from typing import Any

import pytest
from bench_helpers import make_hit, make_occurrence, make_part, make_truth_query
from benchmarks import match


def _codes(result: dict[str, Any]) -> list[str]:
    return [failure["code"] for failure in result["failures"]]


def test_prediction_bbox_returns_union_and_none() -> None:
    assert match.prediction_bbox([]) is None
    assert match.prediction_bbox([[10.0, 20.0, 30.0, 40.0]]) == [10.0, 20.0, 30.0, 40.0]
    assert match.prediction_bbox([[10.0, 20.0, 30.0, 40.0], [5.0, 25.0, 15.0, 35.0]]) == [
        5.0,
        20.0,
        30.0,
        40.0,
    ]


def test_exact_single_hit_is_matched() -> None:
    box = [100.0, 200.0, 120.0, 212.0]
    truth_query = make_truth_query([make_occurrence([make_part(box)])])

    result = match.match_query(
        truth_query,
        [make_hit(0, [box])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["expected_count"] == 1
    assert result["predicted_count"] == 1
    assert result["ambiguous_count"] == 0
    assert result["failures"] == []

    pair = result["matched_pairs"][0]
    assert pair["center_error_pt"] == 0.0
    assert pair["iou"] == 1.0
    assert pair["rect_count_exact"] is True
    assert pair["merged_box_detected"] is False
    assert pair["ambiguous"] is False
    assert pair["line_recall"] == 1.0
    assert pair["line_precision"] == 1.0


def test_same_page_repeats_match_one_to_one_regardless_of_order() -> None:
    first = [100.0, 200.0, 120.0, 212.0]
    second = [200.0, 200.0, 220.0, 212.0]
    truth_query = make_truth_query(
        [
            make_occurrence([make_part(first)], occurrence_id="q0-occ0"),
            make_occurrence([make_part(second)], occurrence_id="q0-occ1"),
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [second]), make_hit(1, [first])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0

    assignment = {
        pair["expected_occurrence_id"]: pair["predicted_index"] for pair in result["matched_pairs"]
    }
    assert assignment == {"q0-occ0": 1, "q0-occ1": 0}


def test_unmatched_prediction_is_a_false_positive() -> None:
    box = [100.0, 200.0, 120.0, 212.0]
    far = [300.0, 200.0, 320.0, 212.0]
    truth_query = make_truth_query([make_occurrence([make_part(box)])])

    result = match.match_query(
        truth_query,
        [make_hit(0, [box]), make_hit(1, [far])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)
    assert _codes(result) == ["F03_EXTRA_HIT"]


def test_missing_prediction_is_a_false_negative() -> None:
    first = [100.0, 200.0, 120.0, 212.0]
    second = [200.0, 200.0, 220.0, 212.0]
    truth_query = make_truth_query(
        [
            make_occurrence([make_part(first)], occurrence_id="q0-occ0"),
            make_occurrence([make_part(second)], occurrence_id="q0-occ1"),
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [first])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 1)
    assert _codes(result) == ["F02_MISSING_OCCURRENCE"]


def test_no_prediction_when_expected_reports_no_hit() -> None:
    box = [100.0, 200.0, 120.0, 212.0]
    truth_query = make_truth_query([make_occurrence([make_part(box)])])

    result = match.match_query(
        truth_query,
        [],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)
    assert _codes(result) == ["F01_NO_HIT"]


def test_zero_expected_and_zero_predicted_has_no_failures() -> None:
    truth_query = make_truth_query([])

    result = match.match_query(
        truth_query,
        [],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 0)
    assert result["failures"] == []
    assert result["page_counts"] == {"expected": [0], "predicted": [0]}


def test_same_count_on_wrong_page_reports_wrong_page() -> None:
    box = [100.0, 200.0, 120.0, 212.0]
    truth_query = make_truth_query([make_occurrence([make_part(box)], page_index=0)])

    result = match.match_query(
        truth_query,
        [make_hit(0, [box], page_index=1)],
        page_count=2,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["page_counts"] == {"expected": [1, 0], "predicted": [0, 1]}
    assert _codes(result) == ["F04_WRONG_PAGE"]


def test_cross_line_parts_pair_vertically_with_one_rect_per_line() -> None:
    upper = [100.0, 240.0, 140.0, 252.0]
    lower = [100.0, 200.0, 140.0, 212.0]
    truth_query = make_truth_query(
        [
            make_occurrence(
                [
                    make_part(lower, line_index=1, text="gamma"),
                    make_part(upper, line_index=0, text="beta"),
                ]
            )
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [lower, upper])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert result["tp"] == 1
    pair = result["matched_pairs"][0]
    assert pair["rect_count_exact"] is True
    assert pair["merged_box_detected"] is False
    assert pair["line_recall"] == 1.0
    assert pair["line_precision"] == 1.0
    assert [entry["text"] for entry in pair["part_matches"]] == ["beta", "gamma"]
    assert pair["part_matches"][0]["rect"] == upper
    assert pair["part_matches"][1]["rect"] == lower
    assert result["failures"] == []


def test_merged_box_spanning_two_lines_is_line_structure_failure() -> None:
    upper = [100.0, 240.0, 140.0, 252.0]
    lower = [100.0, 200.0, 140.0, 212.0]
    merged = [100.0, 200.0, 140.0, 252.0]
    truth_query = make_truth_query(
        [
            make_occurrence(
                [
                    make_part(upper, line_index=0, text="beta"),
                    make_part(lower, line_index=1, text="gamma"),
                ]
            )
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [merged])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert result["tp"] == 1
    pair = result["matched_pairs"][0]
    assert pair["rect_count_exact"] is False
    assert pair["merged_box_detected"] is True
    assert _codes(result) == ["F07_LINE_STRUCTURE"]


def test_near_tie_is_reported_as_ambiguous() -> None:
    expected_box = [100.0, 200.0, 120.0, 212.0]
    exact = [100.0, 200.0, 120.0, 212.0]
    shifted = [100.5, 200.0, 120.5, 212.0]
    truth_query = make_truth_query([make_occurrence([make_part(expected_box)])])

    result = match.match_query(
        truth_query,
        [make_hit(0, [exact]), make_hit(1, [shifted])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["ambiguous_count"] == 1
    assert result["matched_pairs"][0]["ambiguous"] is True
    assert _codes(result) == ["F03_EXTRA_HIT", "F11_AMBIGUOUS_MATCH"]


def test_match_sample_collects_failures_per_query() -> None:
    first = [100.0, 200.0, 120.0, 212.0]
    second = [200.0, 200.0, 220.0, 212.0]
    truth_data = {
        "page_count": 1,
        "queries": [
            {
                "query_index": 0,
                "query": "first",
                "match_case": True,
                "occurrences": [make_occurrence([make_part(first)])],
            },
            {
                "query_index": 1,
                "query": "second",
                "match_case": True,
                "occurrences": [make_occurrence([make_part(second)])],
            },
        ],
    }

    result = match.match_sample(
        truth_data,
        [[make_hit(0, [first])]],
        {"match_radius_pt": 12.0},
    )

    assert len(result["queries"]) == 2
    assert result["queries"][0]["tp"] == 1
    assert result["queries"][1]["tp"] == 0
    assert [failure["code"] for failure in result["failures"]] == ["F01_NO_HIT"]
    assert result["failures"][0]["query"] == "second"


def test_min_cost_matching_avoids_greedy_detection_loss() -> None:
    """Greedy nearest-neighbour matching would under-count true positives.

    Centres are chosen so that the greedy first choice ``E1-P1`` (distance 5)
    blocks both ``E1-P2`` (distance 7) and ``E2-P1`` (distance 6), while
    ``E2-P2`` (distance 16) is outside the 12 pt radius.  Greedy therefore
    reports TP=1/FP=1/FN=1, but a maximum-cardinality, minimum-total-distance
    assignment correctly reports TP=2/FP=0/FN=0.
    """

    e1 = [90.0, 294.0, 110.0, 306.0]  # centre (100, 300)
    e2 = [97.0, 299.65685, 117.0, 311.65685]  # centre (107, 305.65685)
    p1 = [95.0, 294.0, 115.0, 306.0]  # centre (105, 300)
    p2 = [84.55556, 289.60022, 104.55556, 301.60022]  # centre (94.55556, 295.60022)

    truth_query = make_truth_query(
        [
            make_occurrence([make_part(e1)], occurrence_id="q0-occ0"),
            make_occurrence([make_part(e2)], occurrence_id="q0-occ1"),
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [p1]), make_hit(1, [p2])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["ambiguous_count"] == 0
    assignment = {
        pair["expected_occurrence_id"]: pair["predicted_index"] for pair in result["matched_pairs"]
    }
    assert assignment == {"q0-occ0": 1, "q0-occ1": 0}
    assert sorted(pair["center_error_pt"] for pair in result["matched_pairs"]) == pytest.approx(
        [6.0, 7.0]
    )


def test_matching_is_independent_of_input_order() -> None:
    first = [100.0, 200.0, 120.0, 212.0]
    second = [200.0, 200.0, 220.0, 212.0]

    forward_truth = make_truth_query(
        [
            make_occurrence([make_part(first)], occurrence_id="q0-occ0"),
            make_occurrence([make_part(second)], occurrence_id="q0-occ1"),
        ]
    )
    reversed_truth = make_truth_query(
        [
            make_occurrence([make_part(second)], occurrence_id="q0-occ1"),
            make_occurrence([make_part(first)], occurrence_id="q0-occ0"),
        ]
    )

    forward = match.match_query(
        forward_truth,
        [make_hit(0, [first]), make_hit(1, [second])],
        page_count=1,
        match_radius_pt=12.0,
    )
    reversed_result = match.match_query(
        reversed_truth,
        [make_hit(0, [second]), make_hit(1, [first])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (forward["tp"], forward["fp"], forward["fn"]) == (2, 0, 0)
    assert (reversed_result["tp"], reversed_result["fp"], reversed_result["fn"]) == (2, 0, 0)

    forward_assignment = {
        pair["expected_occurrence_id"]: pair["predicted_index"] for pair in forward["matched_pairs"]
    }
    reversed_assignment = {
        pair["expected_occurrence_id"]: pair["predicted_index"]
        for pair in reversed_result["matched_pairs"]
    }
    assert forward_assignment == {"q0-occ0": 0, "q0-occ1": 1}
    assert reversed_assignment == {"q0-occ0": 1, "q0-occ1": 0}


def test_cross_page_partial_match_counts_extra_and_missing() -> None:
    page0_box = [100.0, 200.0, 120.0, 212.0]
    page1_box = [100.0, 300.0, 120.0, 312.0]
    far_page0_box = [400.0, 200.0, 420.0, 212.0]

    truth_query = make_truth_query(
        [
            make_occurrence([make_part(page0_box)], page_index=0, occurrence_id="q0-occ0"),
            make_occurrence([make_part(page1_box)], page_index=1, occurrence_id="q0-occ1"),
        ]
    )

    result = match.match_query(
        truth_query,
        [
            make_hit(0, [page0_box], page_index=0),
            make_hit(1, [far_page0_box], page_index=0),
        ],
        page_count=2,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["page_counts"] == {"expected": [1, 1], "predicted": [2, 0]}
    assert _codes(result) == ["F04_WRONG_PAGE"]

    pair = result["matched_pairs"][0]
    assert pair["expected_occurrence_id"] == "q0-occ0"
    assert pair["center_error_pt"] == 0.0


def test_ambiguity_threshold_is_strictly_less_than_one_point() -> None:
    expected_box = [100.0, 200.0, 120.0, 212.0]
    exact = [100.0, 200.0, 120.0, 212.0]
    one_point_away = [101.0, 200.0, 121.0, 212.0]
    truth_query = make_truth_query([make_occurrence([make_part(expected_box)])])

    result = match.match_query(
        truth_query,
        [make_hit(0, [exact]), make_hit(1, [one_point_away])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["ambiguous_count"] == 0
    assert result["matched_pairs"][0]["ambiguous"] is False
    assert _codes(result) == ["F03_EXTRA_HIT"]


def test_line_parts_use_min_cost_assignment_not_vertical_zip() -> None:
    """Per-line pairing must minimise distance instead of zipping by height.

    A positional top-to-top / bottom-to-bottom zip would pair each part with a
    rectangle about 100.5 pt away.  The optimal assignment pairs them at 90 pt
    each while still assigning exactly one rectangle per expected line.
    """

    top_part = [90.0, 294.0, 110.0, 306.0]  # centre (100, 300)
    bottom_part = [190.0, 194.0, 210.0, 206.0]  # centre (200, 200)
    top_rect = [190.0, 284.0, 210.0, 296.0]  # centre (200, 290)
    bottom_rect = [90.0, 204.0, 110.0, 216.0]  # centre (100, 210)

    truth_query = make_truth_query(
        [
            make_occurrence(
                [
                    make_part(top_part, line_index=0, text="top"),
                    make_part(bottom_part, line_index=1, text="bottom"),
                ]
            )
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [bottom_rect, top_rect])],
        page_count=1,
        match_radius_pt=110.0,
    )

    assert result["tp"] == 1
    pair = result["matched_pairs"][0]
    assert pair["rect_count_exact"] is True
    assert pair["merged_box_detected"] is False
    assert pair["line_recall"] == 1.0
    assert pair["line_precision"] == 1.0
    assert [entry["text"] for entry in pair["part_matches"]] == ["top", "bottom"]
    assert pair["part_matches"][0]["rect"] == bottom_rect
    assert pair["part_matches"][1]["rect"] == top_rect
    assert [entry["center_error_pt"] for entry in pair["part_matches"]] == pytest.approx(
        [90.0, 90.0]
    )


def test_equal_cost_optima_are_reported_as_ambiguous() -> None:
    """Two optimal assignments of equal total cost must not be silently resolved.

    ``E0`` sits 1 pt from ``P0`` and 4 pt from ``P1``; ``E1`` sits 6 pt from
    ``P0`` and 9 pt from ``P1``.  Both perfect matchings cost exactly 10 pt, so
    the evidence cannot decide which prediction corresponds to which
    occurrence.  Detection counts must stay ``2/0/0`` and both occurrences must
    be marked ambiguous (``F11_AMBIGUOUS_MATCH``); only the per-pair geometry is
    excluded from coordinate aggregates.
    """

    e0 = [90.0, 194.0, 110.0, 206.0]  # centre (100, 200)
    e1 = [90.0, 199.0, 110.0, 211.0]  # centre (100, 205)
    p0 = [90.0, 193.0, 110.0, 205.0]  # centre (100, 199)
    p1 = [90.0, 190.0, 110.0, 202.0]  # centre (100, 196)

    truth_query = make_truth_query(
        [
            make_occurrence([make_part(e0)], occurrence_id="q0-occ0"),
            make_occurrence([make_part(e1)], occurrence_id="q0-occ1"),
        ]
    )

    result = match.match_query(
        truth_query,
        [make_hit(0, [p0]), make_hit(1, [p1])],
        page_count=1,
        match_radius_pt=12.0,
    )

    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["ambiguous_count"] == 2
    assert all(pair["ambiguous"] is True for pair in result["matched_pairs"])
    assert _codes(result) == ["F11_AMBIGUOUS_MATCH"]


def test_min_cost_solver_is_invariant_under_row_and_column_order() -> None:
    """Permuting rows/columns must not change cardinality or total cost.

    The same four distances are presented with the expected and predicted
    orders swapped.  A stable solver must still return a two-pair matching of
    total cost 10 pt; which of the two equal-cost pairings is reported is
    irrelevant because ``_tied_rows`` marks both rows ambiguous.
    """

    costs = [[1.0, 4.0], [6.0, 9.0]]
    rows_swapped = [[6.0, 9.0], [1.0, 4.0]]
    columns_swapped = [[4.0, 1.0], [9.0, 6.0]]
    both_swapped = [[9.0, 6.0], [4.0, 1.0]]

    for matrix in (costs, rows_swapped, columns_swapped, both_swapped):
        pairs = match._min_cost_pairs(matrix)
        assert len(pairs) == 2
        assert sum(matrix[row][column] for row, column in pairs) == pytest.approx(10.0)
        assert match._tied_rows(matrix, pairs) == {0, 1}


def test_unique_optimum_is_not_reported_ambiguous() -> None:
    """A strictly better assignment must not be flagged as a tie."""

    costs = [[1.0, 5.0], [5.0, 1.0]]

    pairs = match._min_cost_pairs(costs)

    assert len(pairs) == 2
    assert sum(costs[row][column] for row, column in pairs) == pytest.approx(2.0)
    assert match._tied_rows(costs, pairs) == set()
