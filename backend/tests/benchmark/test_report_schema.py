"""Report structure, JSON safety, and Markdown rendering."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from bench_helpers import DATASET_DIR, make_hit, make_record
from benchmarks import match, metrics
from benchmarks import report as report_module
from benchmarks import truth as truth_module

TOP_LEVEL_KEYS = {
    "benchmark",
    "report_version",
    "generated_at_utc",
    "dataset",
    "environment",
    "summary",
    "samples",
    "failures",
    "notes",
}


def _records(
    specs: list[dict[str, Any]],
    truths: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec, truth_data in zip(specs, truths, strict=True):
        predictions: list[list[dict[str, Any]]] = []
        for query in truth_data["queries"]:
            predictions.append(
                [
                    make_hit(
                        index,
                        [part["box"] for part in occurrence["parts"]],
                        page_index=occurrence["page_index"],
                    )
                    for index, occurrence in enumerate(query["occurrences"])
                ]
            )
        matched = match.match_sample(truth_data, predictions, spec)
        records.append(
            make_record(
                sample_id=spec["id"],
                queries=matched["queries"],
                failures=matched["failures"],
            )
        )
    return records


def _build_report(
    specs: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return report_module.build_report(
        dataset_name="v0",
        specs=specs,
        truths=truths,
        records=records,
        dataset_hash=truth_module.dataset_sha256(DATASET_DIR),
        environment={
            "python": "3.11.13",
            "platform_system": "Linux",
            "platform_machine": "x86_64",
        },
        generated_at="2026-10-19T00:00:00Z",
    )


@pytest.fixture(scope="module")
def dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = truth_module.load_specs(DATASET_DIR)
    truths = [truth_module.compute_truth(spec) for spec in specs]
    return specs, truths


@pytest.fixture(scope="module")
def built(dataset) -> dict[str, Any]:
    specs, truths = dataset
    return _build_report(specs, truths, _records(specs, truths))


def test_report_has_expected_top_level_sections(built: dict[str, Any]) -> None:
    assert set(built) == TOP_LEVEL_KEYS
    assert built["benchmark"] == "rag-anchor-pdf-anchoring"
    assert built["report_version"] == 1
    assert built["generated_at_utc"] == "2026-10-19T00:00:00Z"


def test_dataset_section_matches_manifest(built: dict[str, Any], dataset) -> None:
    specs, _ = dataset
    section = built["dataset"]

    assert section["name"] == "v0"
    assert section["sample_count"] == len(specs) == 12
    assert re.fullmatch(r"[0-9a-f]{64}", section["dataset_sha256"])
    assert section["evaluation_split"] == {
        "direct": [spec["id"] for spec in specs],
        "boundary": [],
        "unsupported": [],
    }


def test_summary_reports_perfect_real_metrics(built: dict[str, Any]) -> None:
    summary = built["summary"]

    assert summary["quality_sample_count"] == 12
    assert summary["excluded_sample_count"] == 0
    assert summary["query_count"] == 12
    assert summary["detection"]["tp"] == 14
    assert summary["detection"]["fp"] == 0
    assert summary["detection"]["fn"] == 0
    assert summary["detection"]["precision"] == 1.0
    assert summary["detection"]["recall"] == 1.0
    assert summary["detection"]["f1"] == 1.0
    assert summary["detection"]["count_match_rate"] == 1.0
    assert summary["page"]["page_count_exact_rate"] == 1.0
    assert summary["geometry"]["occurrence"]["center_error_pt"]["count"] == 14
    assert summary["multiline"]["rect_count_exact_rate"] == 1.0


def test_samples_follow_manifest_order_and_schema(built: dict[str, Any], dataset) -> None:
    specs, _ = dataset
    samples = built["samples"]

    assert [sample["sample_id"] for sample in samples] == [spec["id"] for spec in specs]
    expected_fields = {
        "sample_id",
        "scenario",
        "title",
        "evaluation",
        "truth_level",
        "spec_sha256",
        "page_count",
        "preflight_ok",
        "preflight_errors",
        "engine_error",
        "queries",
        "failures",
        "timing",
    }
    for sample, spec in zip(samples, specs, strict=True):
        assert set(sample) == expected_fields
        assert sample["spec_sha256"] == truth_module.spec_sha256(spec)
        assert sample["preflight_ok"] is True
        assert sample["engine_error"] is False
        assert sample["preflight_errors"] == []


def test_no_failures_for_clean_run(built: dict[str, Any]) -> None:
    failures = built["failures"]

    assert failures["counts"] == {}
    assert failures["details"] == []
    assert failures["invalid_fixtures"] == []
    assert failures["unsupported_expectations"] == []


def test_report_is_json_serializable_without_nan_or_infinity(
    built: dict[str, Any],
) -> None:
    cleaned = metrics.json_safe(built)
    text = json.dumps(cleaned, ensure_ascii=False, allow_nan=False)

    assert "NaN" not in text
    assert "Infinity" not in text
    assert '"quality_sample_count": 12' in text


def test_report_contains_no_local_absolute_paths(built: dict[str, Any]) -> None:
    text = json.dumps(metrics.json_safe(built), ensure_ascii=False)

    assert str(DATASET_DIR) not in text
    assert "/data/wwj" not in text
    assert "/tmp/" not in text
    assert "/root/" not in text


def test_write_json_round_trips(built: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"

    written = report_module.write_json(built, target)

    assert written == target
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    loaded = json.loads(text)
    assert set(loaded) == TOP_LEVEL_KEYS
    assert loaded["summary"]["detection"]["f1"] == 1.0


def test_render_markdown_contains_required_sections(built: dict[str, Any]) -> None:
    markdown = report_module.render_markdown(built)

    assert markdown.startswith("# RAG-Anchor PDF anchoring benchmark")
    for heading in ("## Summary", "## Samples", "## Geometry", "## Failures", "## Notes"):
        assert heading in markdown
    assert "A01" in markdown
    assert "A05" in markdown
    assert "No failures recorded." in markdown
    assert "not a quality gate" in markdown


def test_write_markdown_round_trips(built: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "report.md"

    written = report_module.write_markdown(built, target)

    assert written == target
    assert target.read_text(encoding="utf-8").startswith("# RAG-Anchor PDF anchoring benchmark")


def test_invalid_fixture_is_split_into_its_own_section(dataset) -> None:
    specs, truths = dataset
    records = _records(specs, truths)
    records[0] = make_record(
        sample_id="A01",
        preflight_ok=False,
        failures=[
            {"code": "F10_INVALID_FIXTURE", "message": "fixture preflight failed", "query": None}
        ],
    )

    built = _build_report(specs, truths, records)

    assert built["failures"]["counts"] == {"F10_INVALID_FIXTURE": 1}
    assert len(built["failures"]["invalid_fixtures"]) == 1
    assert built["failures"]["invalid_fixtures"][0]["sample_id"] == "A01"
    assert built["summary"]["quality_sample_count"] == 11
    assert built["summary"]["excluded_sample_count"] == 1
