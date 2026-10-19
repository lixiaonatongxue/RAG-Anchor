"""Command line entry point for the M0-06B1 anchoring benchmark.

Usage (from ``backend/``)::

    uv run python -m benchmarks.cli validate --dataset v0
    uv run python -m benchmarks.cli generate --dataset v0
    uv run python -m benchmarks.cli run --dataset v0 --format json,md
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import generate, match, metrics, report, runner, truth

PACKAGE_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = PACKAGE_ROOT.parent


def dataset_dir(name: str) -> Path:
    """Return the specification directory for a dataset name."""
    return PACKAGE_ROOT / "datasets" / name


def default_output_dir(name: str) -> Path:
    """Return the gitignored runtime output directory for a dataset."""
    return BACKEND_ROOT / ".benchmark-data" / name


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("reportlab", "pypdf", "pypdfium2", "rag-anchor"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": packages,
    }


def _load_specs(name: str) -> tuple[list[dict[str, Any]], list[str]]:
    specs = truth.load_specs(dataset_dir(name))
    errors: list[str] = []
    for spec in specs:
        for message in truth.validate_spec(spec):
            errors.append(f"{spec.get('id', '?')}: {message}")
    return specs, errors


def _cmd_validate(args: argparse.Namespace) -> int:
    specs, errors = _load_specs(args.dataset)
    for spec in specs:
        sample_errors = [line for line in errors if line.startswith(f"{spec['id']}: ")]
        if sample_errors:
            print(f"[FAIL] {spec['id']}")
            for line in sample_errors:
                print(f"       {line}")
        else:
            print(
                f"[ OK ] {spec['id']}  scenario={spec['scenario']} "
                f"evaluation={spec['evaluation']} truth={spec['truth_level']}"
            )
    if errors:
        print(f"\n{len(errors)} specification error(s)", file=sys.stderr)
        return 1
    print(f"\nValidated {len(specs)} sample specification(s)")
    return 0


def _generate_all(
    specs: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[list[Path], dict[str, list[str]]]:
    pdf_dir = output_dir / "pdfs"
    paths = generate.generate_dataset(specs, pdf_dir)
    problems: dict[str, list[str]] = {}
    for spec, path in zip(specs, paths, strict=True):
        errors = generate.preflight_pdf(spec, path)
        if errors:
            problems[str(spec["id"])] = errors
    return paths, problems


def _cmd_generate(args: argparse.Namespace) -> int:
    specs, errors = _load_specs(args.dataset)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1
    output_dir = Path(args.out) if args.out else default_output_dir(args.dataset)
    paths, problems = _generate_all(specs, output_dir)
    for path in paths:
        print(f"[GEN ] {path.name}")
    for sample_id, messages in problems.items():
        print(f"[FAIL] {sample_id}", file=sys.stderr)
        for message in messages:
            print(f"       {message}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} fixture(s) failed preflight", file=sys.stderr)
        return 1
    print(f"\nGenerated {len(paths)} fixture(s) and passed preflight")
    return 0


def _record_for_preflight_failure(
    spec: dict[str, Any],
    messages: list[str],
    repeats: int,
) -> dict[str, Any]:
    return {
        "sample_id": str(spec["id"]),
        "evaluation": str(spec["evaluation"]),
        "preflight_ok": False,
        "preflight_errors": messages,
        "engine_error": False,
        "queries": [],
        "failures": [
            {
                "code": "F10_INVALID_FIXTURE",
                "message": "fixture preflight failed",
                "query": None,
            }
        ],
        "timing": {
            "first_call_ms": None,
            "total_median_ms": None,
            "repeats": int(repeats),
            "warmup": 1,
            "unit": "milliseconds",
        },
    }


def _run_one(
    spec: dict[str, Any],
    pdf_path: Path,
    preflight_errors: list[str],
    *,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    truth_data = truth.compute_truth(spec)
    if preflight_errors:
        record = _record_for_preflight_failure(spec, preflight_errors, repeats)
        return record, truth_data

    run = runner.run_sample(spec, pdf_path, repeats=repeats)
    if run["engine_error"]:
        failures = [
            {
                "code": "F09_ENGINE_ERROR",
                "message": f"engine raised {error_type}",
                "query": query.get("query"),
            }
            for query, error_type in zip(run["queries"], run["error_types"], strict=False)
        ]
        record = {
            "sample_id": str(spec["id"]),
            "evaluation": str(spec["evaluation"]),
            "preflight_ok": True,
            "preflight_errors": [],
            "engine_error": True,
            "queries": run["queries"],
            "failures": failures,
            "timing": run["timing"],
        }
        return record, truth_data

    matched = match.match_sample(
        truth_data,
        [query["predicted_hits"] for query in run["queries"]],
        spec,
    )
    record = {
        "sample_id": str(spec["id"]),
        "evaluation": str(spec["evaluation"]),
        "preflight_ok": True,
        "preflight_errors": [],
        "engine_error": False,
        "queries": matched["queries"],
        "failures": matched["failures"],
        "timing": run["timing"],
    }
    return record, truth_data


def _cmd_run(args: argparse.Namespace) -> int:
    specs, errors = _load_specs(args.dataset)
    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    output_dir = Path(args.out) if args.out else default_output_dir(args.dataset)
    pdf_dir = output_dir / "pdfs"
    paths = generate.generate_dataset(specs, pdf_dir)

    records: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    for spec, path in zip(specs, paths, strict=True):
        preflight_errors = generate.preflight_pdf(spec, path)
        record, truth_data = _run_one(
            spec,
            path,
            preflight_errors,
            repeats=args.repeats,
        )
        records.append(record)
        truths.append(truth_data)
        status = "ok" if record["preflight_ok"] and not record["engine_error"] else "problem"
        print(f"[RUN ] {spec['id']}  {status}")

    report_data = report.build_report(
        dataset_name=args.dataset,
        specs=specs,
        truths=truths,
        records=records,
        dataset_hash=truth.dataset_sha256(dataset_dir(args.dataset)),
        environment=_environment(),
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    formats = {item.strip() for item in args.format.split(",") if item.strip()}
    unknown = formats - {"json", "md"}
    if unknown:
        print(f"unsupported report format(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1

    if "json" in formats:
        json_path = report.write_json(report_data, output_dir / "report.json")
        print(f"[OUT ] {json_path}")
    if "md" in formats:
        markdown_path = report.write_markdown(report_data, output_dir / "report.md")
        print(f"[OUT ] {markdown_path}")

    summary = report_data["summary"]
    detection = summary["detection"]
    print(
        "\nSummary: "
        f"quality_samples={summary['quality_sample_count']} "
        f"precision={metrics.json_safe(detection['precision'])} "
        f"recall={metrics.json_safe(detection['recall'])} "
        f"f1={metrics.json_safe(detection['f1'])}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.cli",
        description="Deterministic PDF evidence anchoring benchmark (M0-06B1).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate dataset specifications without generating PDFs",
        description=(
            "Load every sample specification, check its schema, recount query "
            "occurrences from the specification text, and verify layout truth."
        ),
    )
    validate.add_argument("--dataset", default="v0", help="dataset name (default: v0)")
    validate.set_defaults(func=_cmd_validate)

    generate_parser = subparsers.add_parser(
        "generate",
        help="generate deterministic fixture PDFs and run preflight checks",
        description=(
            "Render every specification with reportlab, rewrite it with pypdf to "
            "apply rotation/CropBox, and check structure plus independently "
            "extracted text with pypdf."
        ),
    )
    generate_parser.add_argument("--dataset", default="v0", help="dataset name (default: v0)")
    generate_parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: backend/.benchmark-data/<dataset>)",
    )
    generate_parser.set_defaults(func=_cmd_generate)

    run_parser = subparsers.add_parser(
        "run",
        help="run the benchmark and write JSON/Markdown reports",
        description=(
            "Generate fixtures, call the production search service, match "
            "predictions one-to-one against independent truth, and write reports. "
            "The exit code reflects harness completion, not anchoring quality."
        ),
    )
    run_parser.add_argument("--dataset", default="v0", help="dataset name (default: v0)")
    run_parser.add_argument(
        "--out",
        default=None,
        help="output directory (default: backend/.benchmark-data/<dataset>)",
    )
    run_parser.add_argument(
        "--format",
        default="json,md",
        help="comma separated report formats: json,md (default: json,md)",
    )
    run_parser.add_argument(
        "--repeats",
        type=int,
        default=runner.DEFAULT_REPEATS,
        help=f"timed repeats per query (default: {runner.DEFAULT_REPEATS})",
    )
    run_parser.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
