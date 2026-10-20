"""Deterministic PDF fixture generation and independent preflight checks.

Fixtures are rendered with reportlab from the JSON specification and rewritten
with pypdf to apply rotation and CropBox.  This module must not import
``pypdfium2`` or ``rag_anchor``.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas

from benchmarks.fonts import ensure_registered
from benchmarks.truth import count_in_text, page_text


class GenerationError(RuntimeError):
    """Raised when a fixture cannot be rendered or rewritten."""


def render_pdf_bytes(spec: dict[str, Any]) -> bytes:
    """Render one specification into deterministic PDF bytes."""
    required_fonts = {
        str(segment["font"])
        for page in spec["pages"]
        for line in page["lines"]
        for segment in line["segments"]
    }
    unavailable = ensure_registered(required_fonts)
    if unavailable:
        raise GenerationError("required benchmark fonts are unavailable: " + ", ".join(unavailable))

    width, height = (float(value) for value in spec["page_size"])
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(width, height), invariant=1)
    for page in spec["pages"]:
        for line in page["lines"]:
            baseline_y = float(line["baseline_y"])
            for segment in line["segments"]:
                pdf.setFont(str(segment["font"]), float(segment["size"]))
                pdf.drawString(float(segment["x"]), baseline_y, str(segment["text"]))
        pdf.showPage()
    pdf.save()
    buffer.seek(0)

    reader = PdfReader(buffer)
    writer = PdfWriter()
    for source_page in reader.pages:
        writer.add_page(source_page)

    rotation = int(spec.get("rotation", 0))
    cropbox = spec.get("cropbox")
    for page in writer.pages:
        if rotation:
            page.rotate(rotation)
        if cropbox is not None:
            page.cropbox = RectangleObject([float(value) for value in cropbox])

    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def generate_sample(spec: dict[str, Any], output_dir: Path) -> Path:
    """Write ``<sample_id>.pdf`` into ``output_dir`` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec['id']}.pdf"
    path.write_bytes(render_pdf_bytes(spec))
    return path


def generate_dataset(specs: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Generate every fixture into ``output_dir``."""
    return [generate_sample(spec, output_dir) for spec in specs]


def _close(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    return abs(float(actual) - float(expected)) <= tolerance


def preflight_pdf(spec: dict[str, Any], path: Path) -> list[str]:
    """Check structure, page geometry, and independently extracted text."""
    errors: list[str] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return [f"{path.name}: cannot be opened by pypdf: {type(exc).__name__}"]

    expected_pages = len(spec["pages"])
    if len(reader.pages) != expected_pages:
        errors.append(f"{path.name}: page count {len(reader.pages)} != {expected_pages}")
        return errors

    expected_rotation = int(spec.get("rotation", 0)) % 360
    expected_cropbox = spec.get("cropbox")
    width, height = (float(value) for value in spec["page_size"])
    for page_index, page in enumerate(reader.pages):
        rotation = int(page.rotation) % 360
        if rotation != expected_rotation:
            errors.append(
                f"{path.name}: page {page_index} rotation {rotation} != {expected_rotation}"
            )
        if expected_cropbox is None:
            media = page.mediabox
            if not (_close(media.width, width) and _close(media.height, height)):
                errors.append(
                    f"{path.name}: page {page_index} media size "
                    f"{float(media.width)}x{float(media.height)} "
                    f"!= {width}x{height}"
                )
        else:
            crop = page.cropbox
            values = (
                float(crop.left),
                float(crop.bottom),
                float(crop.right),
                float(crop.top),
            )
            if not all(
                _close(actual, expected)
                for actual, expected in zip(values, expected_cropbox, strict=False)
            ):
                errors.append(
                    f"{path.name}: page {page_index} cropbox {values} "
                    f"!= {tuple(float(value) for value in expected_cropbox)}"
                )

    for query in spec["queries"]:
        expected_per_page = [0] * expected_pages
        for expected in query["expected"]:
            expected_per_page[int(expected["page_index"])] += 1
        for page_index, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception as exc:
                errors.append(
                    f"{path.name}: page {page_index} text extraction failed: {type(exc).__name__}"
                )
                continue
            observed = count_in_text(
                extracted,
                str(query["text"]),
                bool(query["match_case"]),
            )
            if observed != expected_per_page[page_index]:
                errors.append(
                    f"{path.name}: page {page_index} query {query['text']!r} "
                    f"extracted {observed} occurrence(s), expected "
                    f"{expected_per_page[page_index]}"
                )
            if observed == 0 and expected_per_page[page_index] == 0:
                # Negative control: the independent text extractor must also
                # see no occurrence on this page.
                source_text = page_text(spec, page_index)
                if count_in_text(
                    source_text,
                    str(query["text"]),
                    bool(query["match_case"]),
                ):
                    errors.append(
                        f"{path.name}: negative control disagrees with the "
                        f"specification on page {page_index}"
                    )
    return errors
