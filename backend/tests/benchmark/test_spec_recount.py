"""Independent recounting of query occurrences from specification text."""

from __future__ import annotations

from typing import Any

import pytest
from bench_helpers import DATASET_DIR
from benchmarks import truth


@pytest.fixture(scope="module")
def specs() -> list[dict[str, Any]]:
    return truth.load_specs(DATASET_DIR)


def test_declared_counts_match_specification_text(specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        counts = truth.count_spec_occurrences(spec)
        declared = [len(query["expected"]) for query in spec["queries"]]
        assert counts == declared, spec["id"]
        computed = truth.compute_truth(spec)
        assert [query["expected_count"] for query in computed["queries"]] == declared


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("alpha beta", "alpha beta"),
        ("alpha\nbeta", "alpha beta"),
        ("alpha\t \r\n beta", "alpha beta"),
        ("   ", ""),
        ("", ""),
    ],
)
def test_normalize_text_collapses_whitespace(raw: str, expected: str) -> None:
    assert truth.normalize_text(raw) == expected


def test_count_in_text_is_non_overlapping() -> None:
    assert truth.count_in_text("aaaa", "aa", True) == 2
    assert truth.count_in_text("quokka quokka", "quokka", True) == 2


def test_count_in_text_respects_matching_mode() -> None:
    text = "Quokka QUOKKA quokka"
    assert truth.count_in_text(text, "quokka", False) == 3
    assert truth.count_in_text(text, "quokka", True) == 1


def test_count_in_text_handles_empty_query() -> None:
    assert truth.count_in_text("anything", "  ", True) == 0


def test_page_text_joins_segments_and_lines(specs: list[dict[str, Any]]) -> None:
    a01 = next(spec for spec in specs if spec["id"] == "A01")
    assert truth.page_text(a01, 0) == "The zephyr is here."

    c01 = next(spec for spec in specs if spec["id"] == "C01")
    assert truth.page_text(c01, 0) == "alpha beta\ngamma delta"


def test_cross_line_query_counts_after_whitespace_normalization(
    specs: list[dict[str, Any]],
) -> None:
    c01 = next(spec for spec in specs if spec["id"] == "C01")
    assert truth.count_in_text(truth.page_text(c01, 0), "beta gamma", True) == 1


def test_negative_control_counts_zero(specs: list[dict[str, Any]]) -> None:
    a05 = next(spec for spec in specs if spec["id"] == "A05")
    assert truth.count_spec_occurrences(a05) == [0]
    assert truth.compute_truth(a05)["queries"][0]["expected_count"] == 0


def test_multi_page_counts_are_summed() -> None:
    spec = {
        "pages": [
            {"lines": [{"segments": [{"text": "alpha"}]}]},
            {"lines": [{"segments": [{"text": "alpha alpha"}]}]},
        ],
        "queries": [{"text": "alpha", "match_case": True}],
    }
    assert truth.count_spec_occurrences(spec) == [3]
