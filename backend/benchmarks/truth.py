"""Independent truth computation for the PDF anchoring benchmark.

The module deliberately imports neither ``pypdfium2`` nor ``rag_anchor``.
Detection truth (how many occurrences and on which page) is derived only from
the JSON sample specification.  Layout truth is derived from explicit draw
origins plus reportlab font metrics, never from PDFium character boxes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from reportlab.pdfbase import pdfmetrics

SCHEMA_VERSION = 1
EVALUATIONS = frozenset({"direct", "boundary", "unsupported"})
TRUTH_LEVELS = frozenset({"layout-exact", "layout-band", "detection-only", "boundary"})


class DatasetError(RuntimeError):
    """Raised when the manifest or a sample file cannot be loaded."""


class TruthError(ValueError):
    """Raised when layout truth cannot be derived from a specification."""


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Load and minimally validate ``manifest.json``."""
    path = dataset_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetError(f"manifest not found: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise DatasetError(f"manifest is not valid JSON: {path.name}") from exc
    if not isinstance(data, dict):
        raise DatasetError("manifest must be a JSON object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise DatasetError("unsupported manifest schema_version")
    samples = data.get("samples")
    if not isinstance(samples, list) or not samples:
        raise DatasetError("manifest.samples must be a non-empty list")
    if not all(isinstance(name, str) and name.endswith(".json") for name in samples):
        raise DatasetError("manifest.samples must contain JSON file names")
    return data


def load_specs(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load all sample specifications referenced by the manifest."""
    manifest = load_manifest(dataset_dir)
    specs: list[dict[str, Any]] = []
    for name in manifest["samples"]:
        path = dataset_dir / name
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DatasetError(f"sample spec not found: {name}") from exc
        except json.JSONDecodeError as exc:
            raise DatasetError(f"sample spec is not valid JSON: {name}") from exc
        if not isinstance(spec, dict):
            raise DatasetError(f"sample spec must be a JSON object: {name}")
        specs.append(spec)
    identifiers = [spec.get("id") for spec in specs]
    if len(set(identifiers)) != len(identifiers):
        raise DatasetError("sample ids must be unique")
    return specs


def normalize_text(text: str) -> str:
    """Collapse every whitespace run (including newlines) to one space."""
    return " ".join(text.split())


def count_in_text(text: str, query: str, match_case: bool) -> int:
    """Count non-overlapping occurrences after whitespace normalization."""
    haystack = normalize_text(text)
    needle = normalize_text(query)
    if not needle:
        return 0
    if not match_case:
        haystack = haystack.casefold()
        needle = needle.casefold()
    count = 0
    start = 0
    while True:
        found = haystack.find(needle, start)
        if found < 0:
            return count
        count += 1
        start = found + len(needle)


def page_text(spec: dict[str, Any], page_index: int) -> str:
    """Reconstruct the visible text of one page from the specification."""
    lines = spec["pages"][page_index]["lines"]
    return "\n".join(
        "".join(str(segment["text"]) for segment in line["segments"]) for line in lines
    )


def count_spec_occurrences(spec: dict[str, Any]) -> list[int]:
    """Independently count each query in the specification text."""
    counts: list[int] = []
    for query in spec["queries"]:
        total = 0
        for page_index in range(len(spec["pages"])):
            total += count_in_text(
                page_text(spec, page_index),
                str(query["text"]),
                bool(query["match_case"]),
            )
        counts.append(total)
    return counts


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _layout_box(
    segment: dict[str, Any],
    baseline_y: float,
    char_offset: int,
    char_length: int,
) -> list[float]:
    text = str(segment["text"])
    font = str(segment["font"])
    size = float(segment["size"])
    piece = text[char_offset : char_offset + char_length]
    if not piece:
        raise TruthError("layout part resolves to an empty substring")
    try:
        prefix_width = pdfmetrics.stringWidth(text[:char_offset], font, size)
        piece_width = pdfmetrics.stringWidth(piece, font, size)
        ascent, descent = pdfmetrics.getAscentDescent(font, size)
    except Exception as exc:
        raise TruthError(f"no usable font metrics for {font!r}") from exc
    left = float(segment["x"]) + prefix_width
    return [
        left,
        baseline_y + descent,
        left + piece_width,
        baseline_y + ascent,
    ]


def _union_box(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        raise TruthError("an expected occurrence requires at least one part")
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _box_center(box: list[float]) -> list[float]:
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def compute_truth(spec: dict[str, Any]) -> dict[str, Any]:
    """Build detection and layout truth from one specification.

    The returned structure contains no engine output.  Each occurrence lists
    one layout box per expected text line.
    """
    pages = spec["pages"]
    queries: list[dict[str, Any]] = []
    for query_index, query in enumerate(spec["queries"]):
        query_text = str(query["text"])
        match_case = bool(query["match_case"])
        occurrences: list[dict[str, Any]] = []
        for occurrence_index, expected in enumerate(query["expected"]):
            page_index = int(expected["page_index"])
            if page_index < 0 or page_index >= len(pages):
                raise TruthError("expected.page_index is out of range")
            parts: list[dict[str, Any]] = []
            boxes: list[list[float]] = []
            for part in expected["parts"]:
                line_index = int(part["line_index"])
                segment_index = int(part["segment_index"])
                char_offset = int(part["char_offset"])
                char_length = int(part["char_length"])
                try:
                    line = pages[page_index]["lines"][line_index]
                    segment = line["segments"][segment_index]
                except (IndexError, KeyError) as exc:
                    raise TruthError("expected part references a missing segment") from exc
                text = str(segment["text"])
                if char_offset < 0 or char_length <= 0 or char_offset + char_length > len(text):
                    raise TruthError("expected part character range is invalid")
                part_text = text[char_offset : char_offset + char_length]
                box = _layout_box(
                    segment,
                    float(line["baseline_y"]),
                    char_offset,
                    char_length,
                )
                boxes.append(box)
                parts.append(
                    {
                        "line_index": line_index,
                        "segment_index": segment_index,
                        "char_offset": char_offset,
                        "char_length": char_length,
                        "text": part_text,
                        "box": box,
                    }
                )
            joined = normalize_text("\n".join(str(part["text"]) for part in parts))
            if joined != normalize_text(query_text):
                raise TruthError("expected parts do not spell the query after normalization")
            bbox = _union_box(boxes)
            occurrences.append(
                {
                    "occurrence_id": f"q{query_index}-occ{occurrence_index}",
                    "page_index": page_index,
                    "parts": parts,
                    "bbox": bbox,
                    "center": _box_center(bbox),
                }
            )
        queries.append(
            {
                "query_index": query_index,
                "query": query_text,
                "match_case": match_case,
                "expected_count": len(occurrences),
                "occurrences": occurrences,
            }
        )
    return {
        "sample_id": str(spec["id"]),
        "scenario": str(spec["scenario"]),
        "title": str(spec["title"]),
        "evaluation": str(spec["evaluation"]),
        "truth_level": str(spec["truth_level"]),
        "page_count": len(pages),
        "queries": queries,
    }


def validate_spec(spec: dict[str, Any]) -> list[str]:
    """Return a list of schema and self-consistency errors (empty means OK)."""
    errors: list[str] = []
    if not isinstance(spec, dict):
        return ["spec must be a JSON object"]
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must be 1")
    for key in ("id", "scenario", "title"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            errors.append(f"{key} must be a non-empty string")
    if spec.get("evaluation") not in EVALUATIONS:
        errors.append("evaluation must be direct, boundary, or unsupported")
    if spec.get("truth_level") not in TRUTH_LEVELS:
        errors.append("truth_level is not recognized")

    page_size = spec.get("page_size")
    if (
        not isinstance(page_size, list)
        or len(page_size) != 2
        or not all(_number(value) and float(value) > 0 for value in page_size)
    ):
        errors.append("page_size must be two positive numbers")
    if spec.get("rotation") not in (0, 90, 180, 270):
        errors.append("rotation must be 0, 90, 180, or 270")
    cropbox = spec.get("cropbox")
    if cropbox is not None and (
        not isinstance(cropbox, list)
        or len(cropbox) != 4
        or not all(_number(value) for value in cropbox)
        or float(cropbox[0]) >= float(cropbox[2])
        or float(cropbox[1]) >= float(cropbox[3])
    ):
        errors.append("cropbox must be null or [left, bottom, right, top]")

    pages = spec.get("pages")
    if not isinstance(pages, list) or not pages:
        errors.append("pages must be a non-empty list")
    else:
        for page_index, page in enumerate(pages):
            lines = page.get("lines") if isinstance(page, dict) else None
            if not isinstance(lines, list) or not lines:
                errors.append(f"pages[{page_index}].lines must be non-empty")
                continue
            for line_index, line in enumerate(lines):
                if not isinstance(line, dict) or not _number(line.get("baseline_y")):
                    errors.append(f"pages[{page_index}].lines[{line_index}].baseline_y is invalid")
                    continue
                segments = line.get("segments")
                if not isinstance(segments, list) or not segments:
                    errors.append(f"pages[{page_index}].lines[{line_index}].segments is empty")
                    continue
                for segment_index, segment in enumerate(segments):
                    prefix = f"pages[{page_index}].lines[{line_index}].segments[{segment_index}]"
                    if not isinstance(segment, dict):
                        errors.append(f"{prefix} must be an object")
                        continue
                    if not _number(segment.get("x")):
                        errors.append(f"{prefix}.x is invalid")
                    if not isinstance(segment.get("text"), str) or not segment["text"]:
                        errors.append(f"{prefix}.text must be a non-empty string")
                    if not isinstance(segment.get("font"), str) or not segment["font"]:
                        errors.append(f"{prefix}.font must be a non-empty string")
                    if not _number(segment.get("size")) or float(segment["size"]) <= 0:
                        errors.append(f"{prefix}.size must be positive")

    queries = spec.get("queries")
    if not isinstance(queries, list) or not queries:
        errors.append("queries must be a non-empty list")
    else:
        for query_index, query in enumerate(queries):
            prefix = f"queries[{query_index}]"
            if not isinstance(query, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not isinstance(query.get("text"), str) or not query["text"].strip():
                errors.append(f"{prefix}.text must be a non-empty string")
            if not isinstance(query.get("match_case"), bool):
                errors.append(f"{prefix}.match_case must be a boolean")
            expected = query.get("expected")
            if not isinstance(expected, list):
                errors.append(f"{prefix}.expected must be a list")

    match_radius = spec.get("match_radius_pt")
    if not _number(match_radius) or float(match_radius) <= 0:
        errors.append("match_radius_pt must be positive")
    min_separation = spec.get("min_separation_pt")
    if not _number(min_separation) or float(min_separation) < 0:
        errors.append("min_separation_pt must be non-negative")

    if errors:
        return errors

    try:
        counts = count_spec_occurrences(spec)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"cannot recount specification text: {exc}"]
    for query_index, (query, count) in enumerate(zip(spec["queries"], counts, strict=False)):
        declared = len(query["expected"])
        if declared != count:
            errors.append(
                f"queries[{query_index}] declares {declared} occurrences "
                f"but the specification text contains {count}"
            )

    try:
        truth = compute_truth(spec)
    except (TruthError, KeyError, TypeError, ValueError) as exc:
        return [*errors, f"layout truth cannot be computed: {exc}"]

    minimum = float(spec["min_separation_pt"])
    for query in truth["queries"]:
        centers_by_page: dict[int, list[list[float]]] = {}
        for occurrence in query["occurrences"]:
            centers_by_page.setdefault(occurrence["page_index"], []).append(occurrence["center"])
        for page_index, centers in centers_by_page.items():
            for first in range(len(centers)):
                for second in range(first + 1, len(centers)):
                    gap = math.dist(centers[first], centers[second])
                    if gap < minimum:
                        errors.append(
                            f"query {query['query']!r} has ambiguous centers on "
                            f"page {page_index}: {gap:.3f} pt < {minimum:.3f} pt"
                        )
    return errors


def spec_sha256(spec: dict[str, Any]) -> str:
    """Return a stable SHA-256 of one specification."""
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_sha256(dataset_dir: Path) -> str:
    """Return a stable SHA-256 over the manifest and all referenced specs."""
    manifest = load_manifest(dataset_dir)
    digest = hashlib.sha256()
    digest.update((dataset_dir / "manifest.json").read_bytes())
    for name in sorted(manifest["samples"]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((dataset_dir / name).read_bytes())
    return digest.hexdigest()
