"""Adapter around the production anchoring service.

This is the only benchmark module allowed to import ``rag_anchor``.  It never
computes expected values; it only records what the service returned and how
long each call took.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from pathlib import Path
from typing import Any

from benchmarks import metrics
from rag_anchor.services import search_anchors

DEFAULT_REPEATS = 5
DEFAULT_WARMUP = 1


def _hit_to_dict(index: int, hit: Any) -> dict[str, Any]:
    return {
        "index": index,
        "page_index": int(hit.page_index),
        "page_number": int(hit.page_number),
        "matched_text": str(hit.matched_text),
        "char_start": int(hit.char_start),
        "char_count": int(hit.char_count),
        "rects": [list(rect.as_tuple()) for rect in hit.rects],
    }


def _timed_search(
    pdf_path: Path,
    query: str,
    match_case: bool,
) -> tuple[float, list[dict[str, Any]]]:
    started = time.perf_counter()
    hits = search_anchors(pdf_path, query, match_case=match_case)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, [_hit_to_dict(index, hit) for index, hit in enumerate(hits)]


def run_sample(
    spec: dict[str, Any],
    pdf_path: Path,
    *,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP,
) -> dict[str, Any]:
    """Run every query of one sample through the production service."""
    query_records: list[dict[str, Any]] = []
    engine_error = False
    error_types: list[str] = []
    first_call_total = 0.0
    median_total = 0.0

    for query_index, query in enumerate(spec["queries"]):
        query_text = str(query["text"])
        match_case = bool(query["match_case"])
        try:
            first_ms, hits = _timed_search(pdf_path, query_text, match_case)
        except Exception as exc:
            engine_error = True
            error_types.append(type(exc).__name__)
            query_records.append(
                {
                    "query_index": query_index,
                    "query": query_text,
                    "match_case": match_case,
                    "skipped": True,
                    "engine_error": True,
                    "error_type": type(exc).__name__,
                    "predicted_hits": [],
                }
            )
            continue

        for _ in range(max(0, int(warmup))):
            with contextlib.suppress(Exception):
                _timed_search(pdf_path, query_text, match_case)

        timings: list[float] = []
        for _ in range(max(1, int(repeats))):
            elapsed_ms, _ = _timed_search(pdf_path, query_text, match_case)
            timings.append(elapsed_ms)

        median_ms = statistics.median(timings)
        first_call_total += first_ms
        median_total += median_ms
        query_records.append(
            {
                "query_index": query_index,
                "query": query_text,
                "match_case": match_case,
                "skipped": False,
                "engine_error": False,
                "error_type": None,
                "predicted_hits": hits,
                "timing": {
                    "first_call_ms": first_ms,
                    "median_ms": median_ms,
                    "p95_ms": metrics.percentile(timings, 95.0),
                    "max_ms": max(timings),
                    "repeats": len(timings),
                    "warmup": int(warmup),
                },
            }
        )

    return {
        "sample_id": str(spec["id"]),
        "engine_error": engine_error,
        "error_types": error_types,
        "queries": query_records,
        "timing": {
            "first_call_ms": first_call_total,
            "total_median_ms": median_total if not engine_error else None,
            "repeats": int(repeats),
            "warmup": int(warmup),
            "unit": "milliseconds",
        },
    }
