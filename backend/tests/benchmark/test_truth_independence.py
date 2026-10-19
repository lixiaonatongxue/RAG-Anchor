"""Prove that truth and report code never imports the tested engine."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
from bench_helpers import BENCHMARKS_DIR, DATASET_DIR
from benchmarks import truth
from reportlab.pdfbase import pdfmetrics

FORBIDDEN_ROOTS = frozenset({"pypdfium2", "rag_anchor"})
TRUTH_MODULES = ("truth.py", "generate.py", "match.py", "metrics.py", "report.py")


@pytest.fixture(scope="module")
def specs() -> list[dict[str, Any]]:
    return truth.load_specs(DATASET_DIR)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("name", TRUTH_MODULES)
def test_truth_side_modules_do_not_import_the_engine(name: str) -> None:
    roots = _imported_roots(BENCHMARKS_DIR / name)
    assert roots.isdisjoint(FORBIDDEN_ROOTS), sorted(roots & FORBIDDEN_ROOTS)


def test_runner_is_the_only_engine_adapter() -> None:
    adapters = {
        path.name
        for path in sorted(BENCHMARKS_DIR.glob("*.py"))
        if _imported_roots(path) & FORBIDDEN_ROOTS
    }
    assert adapters == {"runner.py"}


def test_truth_structure_contains_no_engine_fields(specs: list[dict[str, Any]]) -> None:
    a01 = next(spec for spec in specs if spec["id"] == "A01")
    data = truth.compute_truth(a01)
    assert set(data) == {
        "sample_id",
        "scenario",
        "title",
        "evaluation",
        "truth_level",
        "page_count",
        "queries",
    }
    query = data["queries"][0]
    assert set(query) == {
        "query_index",
        "query",
        "match_case",
        "expected_count",
        "occurrences",
    }
    occurrence = query["occurrences"][0]
    assert set(occurrence) == {"occurrence_id", "page_index", "parts", "bbox", "center"}
    part = occurrence["parts"][0]
    assert set(part) == {
        "line_index",
        "segment_index",
        "char_offset",
        "char_length",
        "text",
        "box",
    }
    blob = json.dumps(data)
    assert "char_start" not in blob
    assert "rects" not in blob


def test_layout_truth_uses_draw_origin_and_font_metrics(
    specs: list[dict[str, Any]],
) -> None:
    a01 = next(spec for spec in specs if spec["id"] == "A01")
    segment = a01["pages"][0]["lines"][0]["segments"][1]
    occurrence = truth.compute_truth(a01)["queries"][0]["occurrences"][0]

    width = pdfmetrics.stringWidth("zephyr", "Helvetica", 12.0)
    ascent, descent = pdfmetrics.getAscentDescent("Helvetica", 12.0)
    expected = [124.012, 200.0 + descent, 124.012 + width, 200.0 + ascent]

    assert segment["x"] == pytest.approx(124.012)
    assert occurrence["bbox"] == pytest.approx(expected, abs=1e-9)
    assert occurrence["parts"][0]["box"] == pytest.approx(expected, abs=1e-9)


def test_cross_line_truth_keeps_one_box_per_line(specs: list[dict[str, Any]]) -> None:
    c01 = next(spec for spec in specs if spec["id"] == "C01")
    occurrence = truth.compute_truth(c01)["queries"][0]["occurrences"][0]
    parts = occurrence["parts"]

    assert [part["text"] for part in parts] == ["beta", "gamma"]
    assert len(occurrence["bbox"]) == 4
    assert parts[0]["box"][0] == pytest.approx(
        100.0 + pdfmetrics.stringWidth("alpha ", "Helvetica", 12.0),
        abs=1e-9,
    )
    assert parts[1]["box"][0] == pytest.approx(100.0, abs=1e-9)
    assert parts[0]["box"][1] > parts[1]["box"][3]


def test_accented_prefix_truth_starts_at_segment_origin(
    specs: list[dict[str, Any]],
) -> None:
    b02 = next(spec for spec in specs if spec["id"] == "B02")
    part = truth.compute_truth(b02)["queries"][0]["occurrences"][0]["parts"][0]
    assert part["text"] == "quick"
    assert part["box"][0] == pytest.approx(126.016, abs=1e-9)
