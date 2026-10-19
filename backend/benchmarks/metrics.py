"""Pure geometry and metric helpers for the anchoring benchmark.

This module deliberately has no dependency on PDFium, the production service,
or the filesystem.  Every division is guarded so that an undefined metric is
reported as ``None`` (JSON ``null``) instead of ``NaN`` or ``Infinity``.
"""

from __future__ import annotations

import math
from typing import Any

Box = tuple[float, float, float, float]


def as_box(value: Any) -> Box:
    """Return a four-float ``(left, bottom, right, top)`` tuple."""
    left, bottom, right, top = (float(item) for item in value)
    return (left, bottom, right, top)


def center(box: Any) -> tuple[float, float]:
    """Return the geometric center of a box."""
    left, bottom, right, top = as_box(box)
    return ((left + right) / 2.0, (bottom + top) / 2.0)


def area(box: Any) -> float:
    """Return the non-negative area of a box."""
    left, bottom, right, top = as_box(box)
    return max(0.0, right - left) * max(0.0, top - bottom)


def intersection_area(first: Any, second: Any) -> float:
    """Return the area shared by two axis-aligned boxes."""
    a_left, a_bottom, a_right, a_top = as_box(first)
    b_left, b_bottom, b_right, b_top = as_box(second)
    left = max(a_left, b_left)
    bottom = max(a_bottom, b_bottom)
    right = min(a_right, b_right)
    top = min(a_top, b_top)
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def union_box(boxes: Any) -> Box:
    """Return the smallest box enclosing all input boxes."""
    items = [as_box(box) for box in boxes]
    if not items:
        raise ValueError("union_box requires at least one box")
    return (
        min(box[0] for box in items),
        min(box[1] for box in items),
        max(box[2] for box in items),
        max(box[3] for box in items),
    )


def iou(predicted: Any, expected: Any) -> float:
    """Intersection over union; ``0.0`` when the union has zero area."""
    shared = intersection_area(predicted, expected)
    union = area(predicted) + area(expected) - shared
    return shared / union if union > 0 else 0.0


def distance(first: Any, second: Any) -> float:
    """Euclidean distance between two box centers, in PDF points."""
    first_x, first_y = center(first)
    second_x, second_y = center(second)
    return math.hypot(first_x - second_x, first_y - second_y)


def inside_ratio(predicted: Any, expected: Any) -> float:
    """Share of the predicted box that lies inside the truth box."""
    predicted_area = area(predicted)
    if predicted_area <= 0:
        return 0.0
    return intersection_area(predicted, expected) / predicted_area


def coverage(predicted: Any, expected: Any) -> float:
    """Share of the truth box covered by the predicted box."""
    expected_area = area(expected)
    if expected_area <= 0:
        return 0.0
    return intersection_area(predicted, expected) / expected_area


def safe_div(numerator: float, denominator: float) -> float | None:
    """Divide, returning ``None`` when the denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def width_ratio(predicted: Any, expected: Any) -> float | None:
    """Predicted width divided by truth width."""
    predicted_left, _, predicted_right, _ = as_box(predicted)
    expected_left, _, expected_right, _ = as_box(expected)
    return safe_div(predicted_right - predicted_left, expected_right - expected_left)


def height_ratio(predicted: Any, expected: Any) -> float | None:
    """Predicted height divided by truth height."""
    _, predicted_bottom, _, predicted_top = as_box(predicted)
    _, expected_bottom, _, expected_top = as_box(expected)
    return safe_div(predicted_top - predicted_bottom, expected_top - expected_bottom)


def percentile(values: Any, rank: float) -> float | None:
    """Linearly interpolated percentile; ``None`` for an empty sequence."""
    items = sorted(float(value) for value in values)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    position = (len(items) - 1) * (rank / 100.0)
    low = math.floor(position)
    high = min(low + 1, len(items) - 1)
    fraction = position - low
    return items[low] + (items[high] - items[low]) * fraction


def mean(values: Any) -> float | None:
    """Arithmetic mean, or ``None`` for an empty sequence."""
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def distribution(values: Any) -> dict[str, float | int | None]:
    """Return count/mean/median/p95/min/max for a numeric sequence."""
    items = [float(value) for value in values if value is not None]
    if not items:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(items),
        "mean": mean(items),
        "median": percentile(items, 50.0),
        "p95": percentile(items, 95.0),
        "min": min(items),
        "max": max(items),
    }


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _query_detection(query: dict[str, Any]) -> dict[str, float | int | None]:
    true_positive = int(query.get("tp", 0))
    false_positive = int(query.get("fp", 0))
    false_negative = int(query.get("fn", 0))
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _quality_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("evaluation") == "direct"
        and record.get("preflight_ok") is True
        and not record.get("engine_error")
    ]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample results into micro and macro quality metrics."""
    quality_records = _quality_records(records)
    query_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in quality_records:
        for query in record.get("queries", []):
            if not query.get("skipped"):
                query_pairs.append((record, query))

    detections = [_query_detection(query) for _, query in query_pairs]
    true_positive = sum(int(item["tp"] or 0) for item in detections)
    false_positive = sum(int(item["fp"] or 0) for item in detections)
    false_negative = sum(int(item["fn"] or 0) for item in detections)

    micro_precision = safe_div(true_positive, true_positive + false_positive)
    micro_recall = safe_div(true_positive, true_positive + false_negative)
    if micro_precision is None or micro_recall is None:
        micro_f1 = None
    else:
        micro_f1 = safe_div(
            2.0 * micro_precision * micro_recall,
            micro_precision + micro_recall,
        )

    macro_precision = mean(item["precision"] for item in detections)
    macro_recall = mean(item["recall"] for item in detections)
    macro_f1 = mean(item["f1"] for item in detections)

    count_match = sum(
        1
        for _, query in query_pairs
        if int(query["predicted_count"]) == int(query["expected_count"])
    )
    signed_errors = [
        int(query["predicted_count"]) - int(query["expected_count"]) for _, query in query_pairs
    ]

    pages_with_expected = 0
    exact_pages = 0
    page_confusion: list[dict[str, Any]] = []
    for record, query in query_pairs:
        expected = query.get("page_counts", {}).get("expected", [])
        predicted = query.get("page_counts", {}).get("predicted", [])
        for page_index, (expected_count, predicted_count) in enumerate(
            zip(expected, predicted, strict=False)
        ):
            if expected_count > 0:
                pages_with_expected += 1
                if expected_count == predicted_count:
                    exact_pages += 1
            if expected_count != predicted_count:
                page_confusion.append(
                    {
                        "sample_id": record["sample_id"],
                        "query": query["query"],
                        "page_index": page_index,
                        "expected_count": expected_count,
                        "predicted_count": predicted_count,
                        "deficit_on_page": max(0, expected_count - predicted_count),
                        "surplus_on_page": max(0, predicted_count - expected_count),
                    }
                )

    occurrence_center_errors: list[float] = []
    occurrence_ious: list[float] = []
    occurrence_inside_ratios: list[float] = []
    occurrence_coverages: list[float] = []
    occurrence_width_ratios: list[float] = []
    occurrence_height_ratios: list[float] = []
    part_center_errors: list[float] = []
    part_ious: list[float] = []
    line_recalls: list[float] = []
    line_precisions: list[float] = []
    rect_exact = 0
    rect_pairs = 0
    merged_boxes = 0
    ambiguous_pairs = 0

    for _, query in query_pairs:
        for pair in query.get("matched_pairs", []):
            rect_pairs += 1
            if pair.get("rect_count_exact"):
                rect_exact += 1
            if pair.get("merged_box_detected"):
                merged_boxes += 1
            if pair.get("line_recall") is not None:
                line_recalls.append(float(pair["line_recall"]))
            if pair.get("line_precision") is not None:
                line_precisions.append(float(pair["line_precision"]))
            if pair.get("ambiguous"):
                ambiguous_pairs += 1
                continue
            occurrence_center_errors.append(float(pair["center_error_pt"]))
            occurrence_ious.append(float(pair["iou"]))
            occurrence_inside_ratios.append(float(pair["inside_ratio"]))
            occurrence_coverages.append(float(pair["coverage"]))
            if pair.get("width_ratio") is not None:
                occurrence_width_ratios.append(float(pair["width_ratio"]))
            if pair.get("height_ratio") is not None:
                occurrence_height_ratios.append(float(pair["height_ratio"]))
            for part in pair.get("part_matches", []):
                if part.get("rect") is None:
                    continue
                part_center_errors.append(float(part["center_error_pt"]))
                part_ious.append(float(part["iou"]))

    failure_counts: dict[str, int] = {}
    failure_details: list[dict[str, Any]] = []
    for record in records:
        for failure in record.get("failures", []):
            code = str(failure.get("code", "UNKNOWN"))
            failure_counts[code] = failure_counts.get(code, 0) + 1
            failure_details.append(
                {
                    "sample_id": record["sample_id"],
                    **failure,
                }
            )

    sample_timings = [
        record.get("timing", {}).get("total_median_ms")
        for record in records
        if record.get("timing", {}).get("total_median_ms") is not None
    ]

    return {
        "quality_sample_count": len(quality_records),
        "excluded_sample_count": len(records) - len(quality_records),
        "query_count": len(query_pairs),
        "detection": {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
            "macro": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "n_defined": {
                "precision": sum(item["precision"] is not None for item in detections),
                "recall": sum(item["recall"] is not None for item in detections),
                "f1": sum(item["f1"] is not None for item in detections),
            },
            "count_match_rate": safe_div(count_match, len(query_pairs)),
            "signed_count_error": distribution(signed_errors),
        },
        "page": {
            "pages_with_expected": pages_with_expected,
            "exact_pages": exact_pages,
            "page_count_exact_rate": safe_div(exact_pages, pages_with_expected),
            "confusion_pairs": page_confusion,
        },
        "geometry": {
            "occurrence": {
                "center_error_pt": distribution(occurrence_center_errors),
                "iou": distribution(occurrence_ious),
                "inside_ratio": distribution(occurrence_inside_ratios),
                "coverage": distribution(occurrence_coverages),
                "width_ratio": distribution(occurrence_width_ratios),
                "height_ratio": distribution(occurrence_height_ratios),
            },
            "line_part": {
                "center_error_pt": distribution(part_center_errors),
                "iou": distribution(part_ious),
            },
            "ambiguous_pairs_excluded": ambiguous_pairs,
        },
        "multiline": {
            "matched_occurrences": rect_pairs,
            "rect_count_exact_rate": safe_div(rect_exact, rect_pairs),
            "line_recall": mean(line_recalls),
            "line_precision": mean(line_precisions),
            "merged_box_count": merged_boxes,
        },
        "failures": {
            "counts": failure_counts,
            "details": failure_details,
        },
        "timing": {
            "sample_median_ms": distribution(sample_timings),
            "unit": "milliseconds",
            "repeats_policy": "per query: 1 first call, warmup discarded, repeated timed calls",
        },
    }
