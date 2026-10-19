"""One-to-one matching between predictions and independent truth.

Matching deliberately ignores ``char_start``, ``char_count`` and the exact
``matched_text`` form.  A prediction can only match an expected occurrence on
the same page, and each side is consumed at most once.  A one-to-one
minimum-cost maximum matching maximises the number of matched pairs before
minimising the total distance.  Local near-ties closer than
``AMBIGUITY_DELTA_PT`` and globally equal-cost optimal assignments are both
reported as ambiguous instead of being silently resolved.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from benchmarks import metrics

AMBIGUITY_DELTA_PT = 1.0


def prediction_bbox(rects: Any) -> list[float] | None:
    """Return the union box of predicted line rectangles."""
    boxes = [metrics.as_box(rect) for rect in rects]
    if not boxes:
        return None
    return list(metrics.union_box(boxes))


def _center_y(box: Any) -> float:
    return metrics.center(box)[1]


def _overlaps_vertically(first: Any, second: Any) -> bool:
    first_box = metrics.as_box(first)
    second_box = metrics.as_box(second)
    return min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]) > 0.0


def _min_cost_pairs(
    costs: list[list[float | None]],
) -> list[tuple[int, int]]:
    """Return a maximum-cardinality, minimum-total-distance matching.

    ``costs[row][column]`` is a non-negative distance, or ``None`` when the
    pair is outside the match radius.  The algorithm maximises the number of
    matched pairs first, then minimises the summed distance.  Every row and
    column is consumed at most once, so the result is an optimal one-to-one
    assignment rather than a greedy nearest-neighbour choice.
    """
    row_count = len(costs)
    column_count = len(costs[0]) if row_count else 0
    if row_count == 0 or column_count == 0:
        return []

    source = row_count + column_count
    sink = source + 1
    graph: list[list[int]] = [[] for _ in range(sink + 1)]
    # Edge layout: [to, reverse_index, remaining_capacity, cost].
    edges: list[list[int]] = []
    pair_edges: list[tuple[int, int, int]] = []

    def add_edge(start: int, end: int, capacity: int, cost: int) -> int:
        forward = len(edges)
        edges.append([end, forward + 1, capacity, cost])
        edges.append([start, forward, 0, -cost])
        graph[start].append(forward)
        graph[end].append(forward + 1)
        return forward

    for row in range(row_count):
        add_edge(source, row, 1, 0)
    for row in range(row_count):
        for column in range(column_count):
            value = costs[row][column]
            if value is None:
                continue
            edge_index = add_edge(
                row,
                row_count + column,
                1,
                round(value * 1_000_000),
            )
            pair_edges.append((row, column, edge_index))
    for column in range(column_count):
        add_edge(row_count + column, sink, 1, 0)

    # Successive shortest augmenting paths (Bellman-Ford / SPFA).  Each
    # augmentation raises cardinality by one; taking the shortest path keeps
    # the total distance minimal for that cardinality.  A path exists while
    # an unmatched row can still reach an unmatched column.
    while True:
        distance: list[int | None] = [None] * len(graph)
        previous_edge = [-1] * len(graph)
        in_queue = [False] * len(graph)
        distance[source] = 0
        queue: deque[int] = deque([source])
        in_queue[source] = True
        while queue:
            node = queue.popleft()
            in_queue[node] = False
            node_distance = distance[node]
            if node_distance is None:
                continue
            for edge_index in graph[node]:
                edge = edges[edge_index]
                if edge[2] <= 0:
                    continue
                target = edge[0]
                candidate = node_distance + edge[3]
                target_distance = distance[target]
                if target_distance is None or candidate < target_distance:
                    distance[target] = candidate
                    previous_edge[target] = edge_index
                    if not in_queue[target]:
                        queue.append(target)
                        in_queue[target] = True
        if distance[sink] is None:
            break
        node = sink
        while node != source:
            edge_index = previous_edge[node]
            edge = edges[edge_index]
            edge[2] -= 1
            edges[edge[1]][2] += 1
            node = edges[edge[1]][0]

    matched = [(row, column) for row, column, edge_index in pair_edges if edges[edge_index][2] == 0]
    matched.sort()
    return matched


def _matched_cost_units(
    costs: list[list[float | None]],
    pairs: list[tuple[int, int]],
) -> int:
    """Total integer cost of ``pairs`` on the same scale used by the solver."""
    return sum(round(float(costs[row][column]) * 1_000_000) for row, column in pairs)


def _tied_rows(
    costs: list[list[float | None]],
    matching: list[tuple[int, int]],
) -> set[int]:
    """Return rows whose chosen pair is not part of every optimal matching.

    ``_min_cost_pairs`` returns one maximum-cardinality, minimum-total-distance
    matching, but that matching need not be unique.  When a matched pair can be
    forbidden and still be replaced by an equally optimal alternative, the
    correspondence between that occurrence and its prediction is not fixed by
    the evidence.  Reporting it as a confident match would invent a unique
    correspondence, so the caller marks the occurrence ambiguous instead.

    The assignment is re-solved with one matched pair forbidden; if the same
    cardinality and the same total cost remain attainable, the pair is tied.
    Checking every matched pair is sufficient because any alternative optimum
    differs from the chosen one in at least one pair.
    """
    if not matching:
        return set()
    best_size = len(matching)
    best_total = _matched_cost_units(costs, matching)
    tied: set[int] = set()
    for row, column in matching:
        reduced = [list(cost_row) for cost_row in costs]
        reduced[row][column] = None
        alternative = _min_cost_pairs(reduced)
        same_size = len(alternative) == best_size
        same_total = _matched_cost_units(reduced, alternative) == best_total
        if same_size and same_total:
            tied.add(row)
    return tied


def _assign_parts(
    parts: list[dict[str, Any]],
    rects: list[list[float]],
    radius: float,
) -> tuple[list[dict[str, Any]], int, int | None]:
    """Pair expected line boxes with predicted rectangles optimally."""
    expected_order = sorted(parts, key=lambda part: -_center_y(part["box"]))
    costs: list[list[float | None]] = []
    for part in expected_order:
        row: list[float | None] = []
        for rect in rects:
            gap = metrics.distance(rect, part["box"])
            row.append(gap if gap <= radius else None)
        costs.append(row)
    assignment = dict(_min_cost_pairs(costs))

    assignments: list[dict[str, Any]] = []
    for position, part in enumerate(expected_order):
        chosen = assignment.get(position)
        entry: dict[str, Any] = {
            "line_index": int(part["line_index"]),
            "segment_index": int(part["segment_index"]),
            "text": str(part["text"]),
            "box": list(part["box"]),
            "rect": None,
            "center_error_pt": None,
            "iou": None,
            "inside_ratio": None,
            "coverage": None,
        }
        if chosen is not None:
            entry.update(
                {
                    "rect": list(rects[chosen]),
                    "center_error_pt": metrics.distance(rects[chosen], part["box"]),
                    "iou": metrics.iou(rects[chosen], part["box"]),
                    "inside_ratio": metrics.inside_ratio(rects[chosen], part["box"]),
                    "coverage": metrics.coverage(rects[chosen], part["box"]),
                }
            )
        assignments.append(entry)

    assigned_count = len(assignment)
    line_precision = None if not rects else assigned_count / len(rects)
    return assignments, assigned_count, line_precision


def _ambiguous_expected(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    radius: float,
) -> set[int]:
    ambiguous: set[int] = set()
    for expected_index, occurrence in enumerate(expected):
        distances = sorted(
            metrics.distance(prediction_bbox(hit["rects"]), occurrence["bbox"])
            for hit in predicted
            if prediction_bbox(hit["rects"]) is not None
        )
        distances = [distance for distance in distances if distance <= radius]
        if len(distances) >= 2 and distances[1] - distances[0] < AMBIGUITY_DELTA_PT:
            ambiguous.add(expected_index)
    return ambiguous


def match_query(
    truth_query: dict[str, Any],
    predicted_hits: list[dict[str, Any]],
    *,
    page_count: int,
    match_radius_pt: float,
) -> dict[str, Any]:
    """Match one query's predictions against one query's truth."""
    radius = float(match_radius_pt)
    expected = list(truth_query["occurrences"])
    predicted = list(predicted_hits)

    expected_per_page = [0] * page_count
    predicted_per_page = [0] * page_count
    for occurrence in expected:
        expected_per_page[int(occurrence["page_index"])] += 1
    for hit in predicted:
        predicted_per_page[int(hit["page_index"])] += 1

    by_page: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, hit in enumerate(predicted):
        by_page.setdefault(int(hit["page_index"]), []).append((index, hit))

    matched_candidates: list[tuple[float, int, int]] = []
    ambiguous_indices: set[int] = set()
    for page_index, page_predicted in by_page.items():
        page_expected = [
            (index, occurrence)
            for index, occurrence in enumerate(expected)
            if int(occurrence["page_index"]) == page_index
        ]
        page_ambiguous = _ambiguous_expected(
            [occurrence for _, occurrence in page_expected],
            [hit for _, hit in page_predicted],
            radius,
        )
        ambiguous_indices.update(
            expected_index
            for offset, (expected_index, _) in enumerate(page_expected)
            if offset in page_ambiguous
        )

        page_boxes = [prediction_bbox(hit["rects"]) for _, hit in page_predicted]
        costs: list[list[float | None]] = []
        for _, occurrence in page_expected:
            row: list[float | None] = []
            for box in page_boxes:
                if box is None:
                    row.append(None)
                    continue
                gap = metrics.distance(box, occurrence["bbox"])
                row.append(gap if gap <= radius else None)
            costs.append(row)

        # Optimal one-to-one assignment per page.  Each expected occurrence and
        # each predicted hit belongs to exactly one page, so per-page optimality
        # implies global optimality.
        assignment = _min_cost_pairs(costs)

        # ``_min_cost_pairs`` returns a single optimum, but equal-cost optima may
        # coexist.  When the evidence admits another equally good assignment that
        # maps this occurrence to a different prediction, the correspondence is
        # not determined by the data, so the occurrence is reported ambiguous
        # rather than silently assigned.  Detection counts (TP/FP/FN) are not
        # affected; only the per-pair geometry is excluded.
        tied_rows = _tied_rows(costs, assignment)
        ambiguous_indices.update(
            expected_index
            for offset, (expected_index, _) in enumerate(page_expected)
            if offset in tied_rows
        )

        for local_expected, local_predicted in assignment:
            expected_index = page_expected[local_expected][0]
            predicted_index = page_predicted[local_predicted][0]
            gap = costs[local_expected][local_predicted]
            if gap is not None:
                matched_candidates.append((gap, expected_index, predicted_index))

    # Sort for deterministic report ordering; the assignment itself is already
    # fixed and is independent of this ordering.
    matched_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    pairs: list[dict[str, Any]] = []
    for _, expected_index, predicted_index in matched_candidates:
        occurrence = expected[expected_index]
        hit = predicted[predicted_index]
        rects = [metrics.as_box(rect) for rect in hit["rects"]]
        bbox = prediction_bbox(rects)
        parts = occurrence["parts"]
        part_matches, assigned, line_precision = _assign_parts(parts, rects, radius)
        merged = any(
            sum(_overlaps_vertically(rect, part["box"]) for part in parts) >= 2 for rect in rects
        )
        pairs.append(
            {
                "expected_occurrence_id": str(occurrence["occurrence_id"]),
                "predicted_index": int(hit.get("index", predicted_index)),
                "ambiguous": expected_index in ambiguous_indices,
                "center_error_pt": metrics.distance(bbox, occurrence["bbox"]),
                "iou": metrics.iou(bbox, occurrence["bbox"]),
                "inside_ratio": metrics.inside_ratio(bbox, occurrence["bbox"]),
                "coverage": metrics.coverage(bbox, occurrence["bbox"]),
                "width_ratio": metrics.width_ratio(bbox, occurrence["bbox"]),
                "height_ratio": metrics.height_ratio(bbox, occurrence["bbox"]),
                "rect_count_exact": len(rects) == len(parts),
                "merged_box_detected": merged,
                "line_recall": assigned / len(parts) if parts else None,
                "line_precision": line_precision,
                "part_matches": part_matches,
            }
        )

    true_positive = len(pairs)
    false_positive = len(predicted) - true_positive
    false_negative = len(expected) - true_positive

    failures: list[dict[str, Any]] = []
    if expected and not predicted:
        failures.append({"code": "F01_NO_HIT", "message": "expected hits but none returned"})
    elif 0 < len(predicted) < len(expected):
        failures.append({"code": "F02_MISSING_OCCURRENCE", "message": "fewer hits than expected"})
    if len(predicted) > len(expected):
        failures.append({"code": "F03_EXTRA_HIT", "message": "more hits than expected"})
    if len(predicted) == len(expected) and predicted_per_page != expected_per_page:
        failures.append({"code": "F04_WRONG_PAGE", "message": "hit counts differ per page"})
    if any(pair["merged_box_detected"] or not pair["rect_count_exact"] for pair in pairs):
        failures.append(
            {"code": "F07_LINE_STRUCTURE", "message": "line rectangle structure differs"}
        )
    if any(pair["ambiguous"] for pair in pairs):
        failures.append({"code": "F11_AMBIGUOUS_MATCH", "message": "ambiguous nearest neighbor"})

    return {
        "query_index": int(truth_query["query_index"]),
        "query": str(truth_query["query"]),
        "match_case": bool(truth_query["match_case"]),
        "skipped": False,
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "page_counts": {
            "expected": expected_per_page,
            "predicted": predicted_per_page,
        },
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "matched_pairs": pairs,
        "ambiguous_count": sum(1 for pair in pairs if pair["ambiguous"]),
        "failures": failures,
    }


def match_sample(
    truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Match every query of one sample and collect query-level failures."""
    page_count = int(truth["page_count"])
    radius = float(spec["match_radius_pt"])
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for query_index, truth_query in enumerate(truth["queries"]):
        hits = predictions[query_index] if query_index < len(predictions) else []
        result = match_query(
            truth_query,
            hits,
            page_count=page_count,
            match_radius_pt=radius,
        )
        results.append(result)
        for failure in result["failures"]:
            failures.append({**failure, "query": result["query"]})
    return {"queries": results, "failures": failures}
