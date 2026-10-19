"""JSON and Markdown report rendering.

Reports never contain absolute paths, user names, or host names.  All metric
values come from :mod:`benchmarks.metrics` and undefined values are rendered as
``null`` (JSON) or ``N/A`` (Markdown).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks import metrics, truth


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _ratios(aggregate: dict[str, Any], section: str) -> dict[str, Any]:
    return aggregate["geometry"][section]


def build_report(
    *,
    dataset_name: str,
    specs: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    records: list[dict[str, Any]],
    dataset_hash: str,
    environment: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    """Assemble the machine-readable report structure."""
    aggregate = metrics.aggregate(records)
    records_by_id = {str(record["sample_id"]): record for record in records}
    truths_by_id = {str(item["sample_id"]): item for item in truths}

    samples: list[dict[str, Any]] = []
    split: dict[str, list[str]] = {"direct": [], "boundary": [], "unsupported": []}
    for spec in specs:
        sample_id = str(spec["id"])
        record = records_by_id[sample_id]
        split[str(spec["evaluation"])].append(sample_id)
        samples.append(
            {
                "sample_id": sample_id,
                "scenario": str(spec["scenario"]),
                "title": str(spec["title"]),
                "evaluation": str(spec["evaluation"]),
                "truth_level": str(spec["truth_level"]),
                "spec_sha256": truth.spec_sha256(spec),
                "page_count": int(truths_by_id[sample_id]["page_count"]),
                "preflight_ok": bool(record["preflight_ok"]),
                "preflight_errors": list(record.get("preflight_errors", [])),
                "engine_error": bool(record["engine_error"]),
                "queries": list(record["queries"]),
                "failures": list(record["failures"]),
                "timing": dict(record["timing"]),
            }
        )

    failure_details = list(aggregate["failures"]["details"])
    report = {
        "benchmark": "rag-anchor-pdf-anchoring",
        "report_version": 1,
        "generated_at_utc": generated_at,
        "dataset": {
            "name": dataset_name,
            "dataset_sha256": dataset_hash,
            "sample_count": len(specs),
            "evaluation_split": split,
        },
        "environment": environment,
        "summary": aggregate,
        "samples": samples,
        "failures": {
            "counts": aggregate["failures"]["counts"],
            "details": failure_details,
            "invalid_fixtures": [
                item for item in failure_details if item.get("code") == "F10_INVALID_FIXTURE"
            ],
            "unsupported_expectations": [
                item
                for item in failure_details
                if item.get("code") == "F12_UNSUPPORTED_EXPECTATION"
            ],
        },
        "notes": {
            "truth_independence": (
                "Detection truth comes from the JSON specification. Layout truth "
                "comes from draw origins and reportlab font metrics. PDFium "
                "charboxes are never used to produce expected values."
            ),
            "iou": (
                "IoU compares PDFium ink rectangles with reportlab layout boxes. "
                "It is reported for information only and is not a quality gate."
            ),
            "quality_gate": (
                "M0-06B1 defines no hard quality gate. The run exit code only "
                "reports whether the harness itself completed."
            ),
        },
    }
    return metrics.json_safe(report)


def write_json(report: dict[str, Any], path: Path) -> Path:
    """Write a JSON report that is guaranteed free of NaN and Infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        metrics.json_safe(report),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        sort_keys=False,
    )
    path.write_text(payload + "\n", encoding="utf-8")
    return path


def render_markdown(report: dict[str, Any]) -> str:
    """Render the human-readable Markdown report."""
    summary = report["summary"]
    detection = summary["detection"]
    geometry = summary["geometry"]
    occurrence = geometry["occurrence"]
    lines = [
        "# RAG-Anchor PDF anchoring benchmark",
        "",
        f"- Dataset: `{report['dataset']['name']}`",
        f"- Dataset SHA-256: `{report['dataset']['dataset_sha256']}`",
        f"- Samples: {report['dataset']['sample_count']}",
        f"- Generated (UTC): {report['generated_at_utc']}",
        f"- Python: {report['environment'].get('python')}",
        f"- Platform: {report['environment'].get('platform_system')} "
        f"{report['environment'].get('platform_machine')}",
        "",
        "> Detection truth is independent of the tested engine. Layout truth uses",
        "> reportlab font metrics and draw origins. IoU is reported, not gated.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Quality samples | {summary['quality_sample_count']} |",
        f"| Excluded samples | {summary['excluded_sample_count']} |",
        f"| Queries | {summary['query_count']} |",
        f"| TP / FP / FN | {detection['tp']} / {detection['fp']} / {detection['fn']} |",
        f"| Micro precision | {_fmt(detection['precision'])} |",
        f"| Micro recall | {_fmt(detection['recall'])} |",
        f"| Micro F1 | {_fmt(detection['f1'])} |",
        f"| Macro precision | {_fmt(detection['macro']['precision'])} |",
        f"| Macro recall | {_fmt(detection['macro']['recall'])} |",
        f"| Macro F1 | {_fmt(detection['macro']['f1'])} |",
        f"| Count match rate | {_fmt(detection['count_match_rate'])} |",
        f"| Page count exact rate | {_fmt(summary['page']['page_count_exact_rate'])} |",
        f"| Center error median (pt) | {_fmt(occurrence['center_error_pt']['median'])} |",
        f"| Center error p95 (pt) | {_fmt(occurrence['center_error_pt']['p95'])} |",
        f"| Occurrence IoU median | {_fmt(occurrence['iou']['median'])} |",
        f"| Inside ratio median | {_fmt(occurrence['inside_ratio']['median'])} |",
        f"| Rect count exact rate | {_fmt(summary['multiline']['rect_count_exact_rate'])} |",
        f"| Line recall | {_fmt(summary['multiline']['line_recall'])} |",
        f"| Line precision | {_fmt(summary['multiline']['line_precision'])} |",
        "",
        "## Samples",
        "",
        "| Sample | Scenario | Evaluation | Truth level | Query | Expected | Predicted "
        "| TP | FP | FN | Failures |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for sample in report["samples"]:
        failures = ", ".join(item["code"] for item in sample["failures"]) or "-"
        for query in sample["queries"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(sample["sample_id"]),
                        _cell(sample["scenario"]),
                        _cell(sample["evaluation"]),
                        _cell(sample["truth_level"]),
                        _cell(query.get("query", "-")),
                        _cell(query.get("expected_count", "-")),
                        _cell(query.get("predicted_count", "-")),
                        _cell(query.get("tp", "-")),
                        _cell(query.get("fp", "-")),
                        _cell(query.get("fn", "-")),
                        _cell(failures),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Geometry",
            "",
            "| Sample | Query | Center error (pt) | IoU | Inside ratio | Coverage "
            "| Rect count exact |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sample in report["samples"]:
        for query in sample["queries"]:
            for pair in query.get("matched_pairs", []):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _cell(sample["sample_id"]),
                            _cell(query.get("query", "-")),
                            _fmt(pair.get("center_error_pt")),
                            _fmt(pair.get("iou")),
                            _fmt(pair.get("inside_ratio")),
                            _fmt(pair.get("coverage")),
                            "yes" if pair.get("rect_count_exact") else "no",
                        ]
                    )
                    + " |"
                )

    lines.extend(
        [
            "",
            "## Failures",
            "",
        ]
    )
    if report["failures"]["details"]:
        lines.extend(
            [
                "| Sample | Query | Code | Message |",
                "| --- | --- | --- | --- |",
            ]
        )
        for failure in report["failures"]["details"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(failure.get("sample_id", "-")),
                        _cell(failure.get("query", "-")),
                        _cell(failure.get("code", "-")),
                        _cell(failure.get("message", "-")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No failures recorded.")

    lines.extend(
        [
            "",
            f"- Invalid fixtures (F10): {len(report['failures']['invalid_fixtures'])}",
            f"- Unsupported expectations (F12): "
            f"{len(report['failures']['unsupported_expectations'])}",
            "",
            "## Notes",
            "",
            f"- {report['notes']['truth_independence']}",
            f"- {report['notes']['iou']}",
            f"- {report['notes']['quality_gate']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_markdown(report: dict[str, Any], path: Path) -> Path:
    """Write the Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path
