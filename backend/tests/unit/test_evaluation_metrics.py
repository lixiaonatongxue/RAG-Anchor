"""Unit tests for the aggregate metrics of the offline evaluation script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_semantic_retrieval.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evaluate_semantic_retrieval", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(
    *,
    relevant_count: int,
    literal_hits: int,
    semantic_hits: int,
    status: str | None,
    citations: int,
) -> dict:
    return {
        "relevant_count": relevant_count,
        "literal": {"hits": literal_hits},
        "semantic": {"hits": semantic_hits},
        "qa": {"status": status, "citations": citations},
    }


def test_unanswerable_metrics_separate_candidates_from_claims() -> None:
    metrics = _load_script()
    rows = [
        _row(
            relevant_count=0,
            literal_hits=0,
            semantic_hits=2,
            status="evidence_found",
            citations=2,
        ),
        _row(
            relevant_count=0,
            literal_hits=1,
            semantic_hits=0,
            status="answered",
            citations=1,
        ),
        _row(
            relevant_count=0,
            literal_hits=0,
            semantic_hits=0,
            status="insufficient_evidence",
            citations=0,
        ),
        # Answerable rows must not contribute to the unanswerable aggregate.
        _row(
            relevant_count=1,
            literal_hits=5,
            semantic_hits=5,
            status="answered",
            citations=5,
        ),
    ]

    result = metrics._unanswerable_metrics(rows)

    assert result == {
        "count": 3,
        "literal_with_candidates": 1,
        "literal_candidate_rate": 0.3333,
        "semantic_with_candidates": 1,
        "semantic_candidate_rate": 0.3333,
        "qa_claimed_evidence": 2,
        "qa_claimed_evidence_rate": 0.6667,
        "qa_citations": 3,
    }


def test_unanswerable_metrics_handle_an_empty_group() -> None:
    metrics = _load_script()
    rows = [
        _row(
            relevant_count=2,
            literal_hits=1,
            semantic_hits=1,
            status="evidence_found",
            citations=1,
        )
    ]

    result = metrics._unanswerable_metrics(rows)

    assert result["count"] == 0
    assert result["literal_with_candidates"] == 0
    assert result["semantic_with_candidates"] == 0
    assert result["qa_claimed_evidence"] == 0
    assert result["qa_citations"] == 0
    assert result["literal_candidate_rate"] is None
    assert result["semantic_candidate_rate"] is None
    assert result["qa_claimed_evidence_rate"] is None
