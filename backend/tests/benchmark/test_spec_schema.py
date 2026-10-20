"""Dataset schema, manifest, and self-consistency tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from bench_helpers import DATASET_DIR
from benchmarks import truth

REQUIRED_KEYS = {
    "schema_version",
    "id",
    "scenario",
    "title",
    "evaluation",
    "truth_level",
    "page_size",
    "rotation",
    "cropbox",
    "pages",
    "queries",
    "match_radius_pt",
    "min_separation_pt",
}


@pytest.fixture(scope="module")
def specs() -> list[dict[str, Any]]:
    return truth.load_specs(DATASET_DIR)


def _mutated(specs: list[dict[str, Any]], sample_id: str = "A01") -> dict[str, Any]:
    for spec in specs:
        if spec["id"] == sample_id:
            return copy.deepcopy(spec)
    raise AssertionError(f"missing sample {sample_id}")


def test_manifest_lists_unique_samples() -> None:
    manifest = truth.load_manifest(DATASET_DIR)
    assert manifest["schema_version"] == 1
    assert manifest["name"] == "v0"
    assert len(manifest["samples"]) == 18
    assert len(set(manifest["samples"])) == 18


def test_dataset_loads_expected_samples(specs: list[dict[str, Any]]) -> None:
    assert [spec["id"] for spec in specs] == [
        "A01",
        "H01",
        "C01",
        "B02",
        "A05",
        "E01",
        "E02",
        "F01",
        "F02",
        "F03",
        "A03",
        "F04",
        "H03",
        "B01",
        "C02",
        "C03",
        "B03",
        "B04",
    ]
    assert len({spec["id"] for spec in specs}) == len(specs)


def test_specs_have_required_fields(specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        assert set(spec) >= REQUIRED_KEYS
        assert spec["schema_version"] == 1
        assert spec["evaluation"] in truth.EVALUATIONS
        assert spec["truth_level"] in truth.TRUTH_LEVELS


def test_every_bundled_spec_is_valid(specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        assert truth.validate_spec(spec) == [], spec["id"]


def test_schema_json_documents_the_spec_contract() -> None:
    schema = json.loads((DATASET_DIR / "schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert set(schema["required"]) == REQUIRED_KEYS
    properties = schema["properties"]
    assert properties["schema_version"]["const"] == 1
    assert properties["evaluation"]["enum"] == ["direct", "boundary", "unsupported"]
    assert properties["truth_level"]["enum"] == [
        "layout-exact",
        "layout-band",
        "detection-only",
        "boundary",
    ]
    assert properties["rotation"]["enum"] == [0, 90, 180, 270]
    assert properties["queries"]["minItems"] == 1
    assert properties["pages"]["minItems"] == 1


def test_validate_rejects_missing_required_field(specs: list[dict[str, Any]]) -> None:
    bad = _mutated(specs)
    del bad["id"]
    errors = truth.validate_spec(bad)
    assert any("id must be a non-empty string" in error for error in errors), errors


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("evaluation", "unknown", "evaluation must be"),
        ("truth_level", "nope", "truth_level is not recognized"),
        ("rotation", 45, "rotation must be"),
        ("page_size", [0.0, 792.0], "page_size must be"),
        ("cropbox", [400.0, 100.0, 50.0, 500.0], "cropbox must be"),
        ("match_radius_pt", 0.0, "match_radius_pt must be positive"),
        ("min_separation_pt", -1.0, "min_separation_pt must be non-negative"),
    ],
)
def test_validate_rejects_invalid_fields(
    specs: list[dict[str, Any]],
    field: str,
    value: object,
    fragment: str,
) -> None:
    bad = _mutated(specs)
    bad[field] = value
    errors = truth.validate_spec(bad)
    assert any(fragment in error for error in errors), errors


def test_validate_rejects_declared_count_mismatch(specs: list[dict[str, Any]]) -> None:
    bad = _mutated(specs)
    bad["queries"][0]["expected"] = []
    errors = truth.validate_spec(bad)
    assert any("declares 0 occurrences" in error for error in errors), errors


def test_validate_rejects_wrong_part_spelling(specs: list[dict[str, Any]]) -> None:
    bad = _mutated(specs)
    bad["queries"][0]["expected"][0]["parts"][0]["char_length"] = 5
    errors = truth.validate_spec(bad)
    assert any("do not spell the query" in error for error in errors), errors


def test_validate_rejects_out_of_range_page(specs: list[dict[str, Any]]) -> None:
    bad = _mutated(specs)
    bad["queries"][0]["expected"][0]["page_index"] = 5
    errors = truth.validate_spec(bad)
    assert any("page_index is out of range" in error for error in errors), errors


def test_validate_rejects_missing_segment(specs: list[dict[str, Any]]) -> None:
    bad = _mutated(specs)
    bad["queries"][0]["expected"][0]["parts"][0]["segment_index"] = 9
    errors = truth.validate_spec(bad)
    assert any("missing segment" in error for error in errors), errors


def test_validate_rejects_ambiguous_centers(specs: list[dict[str, Any]]) -> None:
    bad = _mutated(specs, "H01")
    bad["pages"][0]["lines"][0]["segments"][2]["x"] = 100.0
    errors = truth.validate_spec(bad)
    assert any("ambiguous centers" in error for error in errors), errors


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(truth.DatasetError, match="manifest not found"):
        truth.load_manifest(tmp_path)


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(truth.DatasetError, match="not valid JSON"):
        truth.load_manifest(tmp_path)


def test_load_specs_rejects_duplicate_ids(specs: list[dict[str, Any]], tmp_path: Path) -> None:
    payload = json.dumps(specs[0])
    (tmp_path / "a.json").write_text(payload, encoding="utf-8")
    (tmp_path / "b.json").write_text(payload, encoding="utf-8")
    manifest = {"schema_version": 1, "samples": ["a.json", "b.json"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(truth.DatasetError, match="ids must be unique"):
        truth.load_specs(tmp_path)
