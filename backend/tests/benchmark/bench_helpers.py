"""Shared builders for benchmark harness tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
DATASET_DIR = BENCHMARKS_DIR / "datasets" / "v0"


def make_hit(
    index: int,
    rects: list[list[float]],
    *,
    page_index: int = 0,
) -> dict[str, Any]:
    """Build the dictionary shape emitted by ``benchmarks.runner``."""
    return {
        "index": index,
        "page_index": page_index,
        "page_number": page_index + 1,
        "matched_text": "x",
        "char_start": index,
        "char_count": 1,
        "rects": rects,
    }


def make_part(
    box: list[float],
    *,
    line_index: int = 0,
    segment_index: int = 0,
    text: str = "x",
) -> dict[str, Any]:
    """Build one expected layout part."""
    return {
        "line_index": line_index,
        "segment_index": segment_index,
        "char_offset": 0,
        "char_length": len(text),
        "text": text,
        "box": box,
    }


def make_occurrence(
    parts: list[dict[str, Any]],
    *,
    page_index: int = 0,
    occurrence_id: str = "q0-occ0",
) -> dict[str, Any]:
    """Build one expected occurrence with a derived union box."""
    boxes = [part["box"] for part in parts]
    bbox = [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]
    return {
        "occurrence_id": occurrence_id,
        "page_index": page_index,
        "parts": parts,
        "bbox": bbox,
        "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
    }


def make_truth_query(
    occurrences: list[dict[str, Any]],
    *,
    query: str = "x",
    match_case: bool = True,
    query_index: int = 0,
) -> dict[str, Any]:
    """Build the dictionary shape produced by ``benchmarks.truth``."""
    return {
        "query_index": query_index,
        "query": query,
        "match_case": match_case,
        "expected_count": len(occurrences),
        "occurrences": occurrences,
    }


def make_record(
    *,
    sample_id: str = "T01",
    evaluation: str = "direct",
    queries: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
    preflight_ok: bool = True,
    engine_error: bool = False,
    total_median_ms: float | None = 10.0,
) -> dict[str, Any]:
    """Build one per-sample record as consumed by ``metrics.aggregate``."""
    return {
        "sample_id": sample_id,
        "evaluation": evaluation,
        "preflight_ok": preflight_ok,
        "preflight_errors": [],
        "engine_error": engine_error,
        "queries": queries or [],
        "failures": failures or [],
        "timing": {
            "first_call_ms": 1.0,
            "total_median_ms": total_median_ms,
            "repeats": 1,
            "warmup": 0,
            "unit": "milliseconds",
        },
    }
