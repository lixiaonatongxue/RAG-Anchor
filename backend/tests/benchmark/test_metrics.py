"""Geometry helpers and aggregate metric composition."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from bench_helpers import make_record
from benchmarks import metrics


def _query(
    *,
    query: str = "x",
    skipped: bool = False,
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    expected_count: int | None = None,
    predicted_count: int | None = None,
    expected_pages: list[int] | None = None,
    predicted_pages: list[int] | None = None,
    pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if expected_count is None:
        expected_count = tp + fn
    if predicted_count is None:
        predicted_count = tp + fp
    return {
        "query": query,
        "skipped": skipped,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "expected_count": expected_count,
        "predicted_count": predicted_count,
        "page_counts": {
            "expected": expected_pages if expected_pages is not None else [expected_count],
            "predicted": predicted_pages if predicted_pages is not None else [predicted_count],
        },
        "matched_pairs": pairs or [],
    }


def _pair(
    *,
    ambiguous: bool = False,
    rect_count_exact: bool = True,
    merged: bool = False,
    line_recall: float = 1.0,
    line_precision: float = 1.0,
    parts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "expected_occurrence_id": "occ",
        "predicted_index": 0,
        "ambiguous": ambiguous,
        "center_error_pt": 0.5,
        "iou": 0.9,
        "inside_ratio": 0.95,
        "coverage": 0.9,
        "width_ratio": 1.0,
        "height_ratio": 1.0,
        "rect_count_exact": rect_count_exact,
        "merged_box_detected": merged,
        "line_recall": line_recall,
        "line_precision": line_precision,
        "part_matches": parts or [],
    }


def _part() -> dict[str, Any]:
    return {
        "rect": [0.0, 0.0, 10.0, 10.0],
        "center_error_pt": 0.25,
        "iou": 0.8,
        "inside_ratio": 0.9,
        "coverage": 0.85,
    }


def test_as_box_and_center() -> None:
    assert metrics.as_box([1, 2, 3, 4]) == (1.0, 2.0, 3.0, 4.0)
    assert metrics.center([0.0, 0.0, 10.0, 20.0]) == (5.0, 10.0)


def test_area_is_never_negative() -> None:
    assert metrics.area([0.0, 0.0, 10.0, 10.0]) == 100.0
    assert metrics.area([10.0, 0.0, 0.0, 10.0]) == 0.0


def test_intersection_and_union() -> None:
    first = [0.0, 0.0, 10.0, 10.0]
    second = [5.0, 0.0, 15.0, 10.0]
    assert metrics.intersection_area(first, second) == 50.0
    assert metrics.intersection_area(first, [20.0, 20.0, 30.0, 30.0]) == 0.0
    assert metrics.union_box([first, second]) == (0.0, 0.0, 15.0, 10.0)


def test_union_box_requires_at_least_one_box() -> None:
    with pytest.raises(ValueError, match="at least one box"):
        metrics.union_box([])


def test_iou_cases() -> None:
    box = [0.0, 0.0, 10.0, 10.0]
    assert metrics.iou(box, box) == 1.0
    assert metrics.iou(box, [20.0, 20.0, 30.0, 30.0]) == 0.0
    assert metrics.iou([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]) == 0.0
    assert metrics.iou([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0]) == pytest.approx(1 / 3)


def test_distance_uses_box_centers() -> None:
    assert metrics.distance([0.0, 0.0, 10.0, 10.0], [-5.0, -5.0, 15.0, 15.0]) == 0.0
    assert metrics.distance([0.0, 0.0, 2.0, 2.0], [3.0, 4.0, 5.0, 6.0]) == pytest.approx(5.0)


def test_inside_ratio_and_coverage() -> None:
    predicted = [2.0, 2.0, 8.0, 8.0]
    expected = [0.0, 0.0, 10.0, 10.0]
    assert metrics.inside_ratio(predicted, expected) == 1.0
    assert metrics.coverage(predicted, expected) == pytest.approx(0.36)
    assert metrics.inside_ratio([0.0, 0.0, 0.0, 0.0], expected) == 0.0
    assert metrics.coverage(predicted, [0.0, 0.0, 0.0, 0.0]) == 0.0


def test_safe_div_and_size_ratios() -> None:
    assert metrics.safe_div(3.0, 2.0) == 1.5
    assert metrics.safe_div(1.0, 0.0) is None
    assert metrics.width_ratio([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 10.0]) == 0.5
    assert metrics.width_ratio([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 5.0, 10.0]) is None
    assert metrics.height_ratio([0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 20.0]) == 0.5
    assert metrics.height_ratio([0.0, 0.0, 10.0, 10.0], [0.0, 5.0, 10.0, 5.0]) is None


def test_percentile_interpolates_and_handles_empty() -> None:
    assert metrics.percentile([], 50.0) is None
    assert metrics.percentile([5.0], 95.0) == 5.0
    assert metrics.percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert metrics.percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)


def test_mean_skips_missing_values() -> None:
    assert metrics.mean([1.0, None, 3.0]) == 2.0
    assert metrics.mean([]) is None
    assert metrics.mean([None, None]) is None


def test_distribution_shape_and_empty_values() -> None:
    assert metrics.distribution([]) == {
        "count": 0,
        "mean": None,
        "median": None,
        "p95": None,
        "min": None,
        "max": None,
    }
    result = metrics.distribution([1.0, 2.0, None, 3.0])
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0


def test_json_safe_replaces_non_finite_floats() -> None:
    cleaned = metrics.json_safe(
        {
            "nan": math.nan,
            "inf": [math.inf, -math.inf, 1.5],
            "nested": {"value": -math.inf},
            "text": "ok",
        }
    )

    assert cleaned == {
        "nan": None,
        "inf": [None, None, 1.5],
        "nested": {"value": None},
        "text": "ok",
    }
    json.dumps(cleaned, allow_nan=False)


def test_aggregate_perfect_detection() -> None:
    record = make_record(queries=[_query(tp=2, pairs=[_pair(), _pair()])])

    result = metrics.aggregate([record])

    assert result["quality_sample_count"] == 1
    assert result["excluded_sample_count"] == 0
    assert result["query_count"] == 1
    assert result["detection"]["tp"] == 2
    assert result["detection"]["fp"] == 0
    assert result["detection"]["fn"] == 0
    assert result["detection"]["precision"] == 1.0
    assert result["detection"]["recall"] == 1.0
    assert result["detection"]["f1"] == 1.0
    assert result["detection"]["macro"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert result["detection"]["n_defined"] == {"precision": 1, "recall": 1, "f1": 1}
    assert result["detection"]["count_match_rate"] == 1.0
    assert result["detection"]["signed_count_error"]["count"] == 1
    assert result["detection"]["signed_count_error"]["mean"] == 0.0
    assert result["page"]["pages_with_expected"] == 1
    assert result["page"]["exact_pages"] == 1
    assert result["page"]["page_count_exact_rate"] == 1.0
    assert result["page"]["confusion_pairs"] == []
    assert result["geometry"]["occurrence"]["center_error_pt"]["count"] == 2
    assert result["geometry"]["occurrence"]["center_error_pt"]["median"] == pytest.approx(0.5)
    assert result["geometry"]["ambiguous_pairs_excluded"] == 0
    assert result["multiline"]["matched_occurrences"] == 2
    assert result["multiline"]["rect_count_exact_rate"] == 1.0
    assert result["multiline"]["line_recall"] == 1.0
    assert result["multiline"]["line_precision"] == 1.0
    assert result["multiline"]["merged_box_count"] == 0
    assert result["failures"] == {"counts": {}, "details": []}
    assert result["timing"]["sample_median_ms"]["count"] == 1
    assert result["timing"]["unit"] == "milliseconds"


def test_aggregate_zero_expected_and_zero_predicted_is_undefined() -> None:
    result = metrics.aggregate([make_record(queries=[_query()])])

    detection = result["detection"]
    assert (detection["tp"], detection["fp"], detection["fn"]) == (0, 0, 0)
    assert detection["precision"] is None
    assert detection["recall"] is None
    assert detection["f1"] is None
    assert detection["macro"] == {"precision": None, "recall": None, "f1": None}
    assert detection["n_defined"] == {"precision": 0, "recall": 0, "f1": 0}
    assert detection["count_match_rate"] == 1.0
    assert result["page"]["pages_with_expected"] == 0
    assert result["page"]["page_count_exact_rate"] is None
    assert result["geometry"]["occurrence"]["center_error_pt"]["count"] == 0


def test_aggregate_missing_prediction_keeps_recall_defined() -> None:
    result = metrics.aggregate([make_record(queries=[_query(tp=0, fn=1)])])

    detection = result["detection"]
    assert detection["precision"] is None
    assert detection["recall"] == 0.0
    assert detection["f1"] is None
    assert detection["n_defined"] == {"precision": 0, "recall": 1, "f1": 0}
    assert detection["macro"]["precision"] is None
    assert detection["macro"]["recall"] == 0.0
    assert detection["signed_count_error"]["mean"] == -1.0


def test_aggregate_extra_prediction_counts_page_surplus() -> None:
    result = metrics.aggregate(
        [
            make_record(
                queries=[
                    _query(
                        tp=0,
                        fp=1,
                        expected_count=0,
                        predicted_count=1,
                        expected_pages=[0],
                        predicted_pages=[1],
                    )
                ]
            )
        ]
    )

    detection = result["detection"]
    assert detection["precision"] == 0.0
    assert detection["recall"] is None
    assert detection["f1"] is None
    assert detection["signed_count_error"]["mean"] == 1.0
    assert result["page"]["confusion_pairs"] == [
        {
            "sample_id": "T01",
            "query": "x",
            "page_index": 0,
            "expected_count": 0,
            "predicted_count": 1,
            "deficit_on_page": 0,
            "surplus_on_page": 1,
        }
    ]


def test_aggregate_count_match_does_not_imply_page_match() -> None:
    result = metrics.aggregate(
        [
            make_record(
                queries=[
                    _query(
                        tp=1,
                        expected_pages=[1, 0],
                        predicted_pages=[0, 1],
                    )
                ]
            )
        ]
    )

    assert result["detection"]["count_match_rate"] == 1.0
    assert result["page"]["pages_with_expected"] == 1
    assert result["page"]["exact_pages"] == 0
    assert result["page"]["page_count_exact_rate"] == 0.0
    assert len(result["page"]["confusion_pairs"]) == 2


def test_aggregate_excludes_non_quality_records() -> None:
    records = [
        make_record(sample_id="G1", queries=[_query(tp=1)]),
        make_record(sample_id="B1", evaluation="boundary", queries=[_query(tp=1)]),
        make_record(sample_id="F1", preflight_ok=False, queries=[_query(tp=1)]),
        make_record(sample_id="E1", engine_error=True, queries=[_query(tp=1)]),
    ]

    result = metrics.aggregate(records)

    assert result["quality_sample_count"] == 1
    assert result["excluded_sample_count"] == 3
    assert result["query_count"] == 1
    assert result["detection"]["tp"] == 1


def test_aggregate_skips_skipped_queries() -> None:
    result = metrics.aggregate([make_record(queries=[_query(skipped=True, tp=5), _query(tp=1)])])

    assert result["query_count"] == 1
    assert result["detection"]["tp"] == 1


def test_aggregate_excludes_ambiguous_pairs_from_geometry() -> None:
    pair = _pair(ambiguous=True, parts=[_part()])

    result = metrics.aggregate([make_record(queries=[_query(tp=1, pairs=[pair])])])

    assert result["geometry"]["ambiguous_pairs_excluded"] == 1
    assert result["geometry"]["occurrence"]["center_error_pt"]["count"] == 0
    assert result["geometry"]["line_part"]["center_error_pt"]["count"] == 0
    assert result["multiline"]["matched_occurrences"] == 1
    assert result["multiline"]["rect_count_exact_rate"] == 1.0


def test_aggregate_counts_merged_boxes() -> None:
    pair = _pair(rect_count_exact=False, merged=True)

    result = metrics.aggregate([make_record(queries=[_query(tp=1, pairs=[pair])])])

    assert result["multiline"]["merged_box_count"] == 1
    assert result["multiline"]["rect_count_exact_rate"] == 0.0


def test_aggregate_counts_line_part_geometry() -> None:
    pair = _pair(parts=[_part(), _part()])

    result = metrics.aggregate([make_record(queries=[_query(tp=1, pairs=[pair])])])

    assert result["geometry"]["line_part"]["center_error_pt"]["count"] == 2
    assert result["geometry"]["line_part"]["iou"]["mean"] == pytest.approx(0.8)


def test_aggregate_collects_failures_from_every_record() -> None:
    record = make_record(
        evaluation="boundary",
        failures=[
            {"code": "F01_NO_HIT", "message": "first"},
            {"code": "F01_NO_HIT", "message": "second"},
            {"code": "F03_EXTRA_HIT", "message": "third"},
        ],
    )

    result = metrics.aggregate([record])

    assert result["failures"]["counts"] == {"F01_NO_HIT": 2, "F03_EXTRA_HIT": 1}
    assert len(result["failures"]["details"]) == 3
    assert result["failures"]["details"][0]["sample_id"] == "T01"


def test_aggregate_timing_distribution() -> None:
    records = [
        make_record(sample_id="S1", total_median_ms=10.0),
        make_record(sample_id="S2", total_median_ms=20.0),
    ]

    result = metrics.aggregate(records)

    timing = result["timing"]["sample_median_ms"]
    assert timing["count"] == 2
    assert timing["mean"] == 15.0
    assert timing["median"] == 15.0
    assert timing["p95"] == pytest.approx(19.5)
    assert timing["min"] == 10.0
    assert timing["max"] == 20.0
