"""Fixture generation, determinism, and independent preflight checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from bench_helpers import DATASET_DIR
from benchmarks import generate, truth
from pypdf import PdfReader
from pypdfium2 import PdfDocument
from pypdfium2.raw import FPDFText_CountChars, FPDFText_GetUnicode

from rag_anchor.services import search_anchors


@pytest.fixture(scope="module")
def specs() -> list[dict[str, Any]]:
    return truth.load_specs(DATASET_DIR)


@pytest.fixture(scope="module")
def generated(specs, tmp_path_factory):
    output = tmp_path_factory.mktemp("generated-pdfs")
    paths = generate.generate_dataset(specs, output)
    return list(zip(specs, paths, strict=True))


def test_every_fixture_passes_preflight(generated) -> None:
    assert len(generated) == 18
    for spec, path in generated:
        assert path.name == f"{spec['id']}.pdf"
        assert path.read_bytes().startswith(b"%PDF-")
        assert generate.preflight_pdf(spec, path) == [], spec["id"]


def test_render_is_deterministic(specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        first = generate.render_pdf_bytes(spec)
        second = generate.render_pdf_bytes(spec)
        assert first == second, spec["id"]


def test_pdf_structure_matches_specification(generated) -> None:
    spec, path = generated[0]
    reader = PdfReader(str(path))
    assert len(reader.pages) == len(spec["pages"])
    page = reader.pages[0]
    assert int(page.rotation) % 360 == spec["rotation"]
    assert float(page.mediabox.width) == pytest.approx(spec["page_size"][0])
    assert float(page.mediabox.height) == pytest.approx(spec["page_size"][1])


def test_rotation_and_cropbox_round_trip(
    specs: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    spec = copy.deepcopy(next(item for item in specs if item["id"] == "A01"))
    spec["rotation"] = 90
    spec["cropbox"] = [50.0, 100.0, 400.0, 500.0]

    path = generate.generate_sample(spec, tmp_path)
    assert generate.preflight_pdf(spec, path) == []

    page = PdfReader(str(path)).pages[0]
    assert int(page.rotation) % 360 == 90
    crop = page.cropbox
    values = [float(crop.left), float(crop.bottom), float(crop.right), float(crop.top)]
    assert values == pytest.approx([50.0, 100.0, 400.0, 500.0])


def test_preflight_detects_page_count_mismatch(specs, generated) -> None:
    spec = copy.deepcopy(next(item for item in specs if item["id"] == "A01"))
    path = next(path for item, path in generated if item["id"] == "A01")
    spec["pages"].append(copy.deepcopy(spec["pages"][0]))

    errors = generate.preflight_pdf(spec, path)
    assert len(errors) == 1
    assert "page count 1 != 2" in errors[0]


def test_preflight_reports_unreadable_file(
    specs: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    spec = next(item for item in specs if item["id"] == "A01")
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a pdf")

    errors = generate.preflight_pdf(spec, path)
    assert errors
    assert "cannot be opened by pypdf" in errors[0]


def test_negative_control_has_no_extractable_query(generated) -> None:
    spec, path = next((item, path) for item, path in generated if item["id"] == "A05")
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "zzzznotfound" not in text
    assert truth.count_in_text(text, spec["queries"][0]["text"], True) == 0


def test_generate_sample_creates_nested_directory(
    specs: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    spec = next(item for item in specs if item["id"] == "A01")
    output = tmp_path / "nested" / "pdfs"
    path = generate.generate_sample(spec, output)
    assert path == output / "A01.pdf"
    assert path.is_file()


def _find(generated, sample_id: str) -> tuple[dict[str, Any], Path]:
    return next((spec, path) for spec, path in generated if spec["id"] == sample_id)


def _pdfium_page_text(path: Path) -> str:
    document = PdfDocument(str(path))
    try:
        page = document[0]
        text_page = page.get_textpage()
        try:
            count = FPDFText_CountChars(text_page)
            return "".join(chr(FPDFText_GetUnicode(text_page, index)) for index in range(count))
        finally:
            text_page.close()
    finally:
        document.close()


def test_b01_cjk_text_is_extracted_consistently(generated) -> None:
    spec, path = _find(generated, "B01")
    reader = PdfReader(str(path))
    pypdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert "中文" in pypdf_text
    assert "证据" in pypdf_text
    assert _pdfium_page_text(path) == "中文 quick 证据"
    assert generate.preflight_pdf(spec, path) == []


def test_b01_production_service_answers_cjk_queries(generated) -> None:
    _, path = _find(generated, "B01")

    for query in ("中文", "证据", "中文 quick", "quick 证据"):
        hits = search_anchors(path, query)
        assert len(hits) == 1, query
        assert hits[0].page_index == 0
    # The benchmark specification still declares only the ASCII query.  The
    # assertions above prove extraction/query support for this fixture only.
    assert search_anchors(path, "quick")[0].matched_text == "quick"


def test_nonbmp_fixtures_do_not_contain_true_u1d11e(generated) -> None:
    for sample_id in ("B03", "B04"):
        spec, path = _find(generated, sample_id)
        reader = PdfReader(str(path))
        pypdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
        pdfium_text = _pdfium_page_text(path)

        # reportlab 5.0.1 writes an odd-length ToUnicode entry (1D11E), which
        # pypdf rejects and PDFium truncates to U+1D11.  Neither extractor sees
        # the original non-BMP code point, so preflight cannot validate it.
        assert "\U0001d11e" not in pypdf_text
        assert "\U0001d11e" not in pdfium_text
        assert "\u1d11" not in pypdf_text
        assert "\u1d11" in pdfium_text
        assert generate.preflight_pdf(spec, path) == []

        # PDFium search ignores whitespace differences (U+1D11 + "quick"
        # matches U+1D11 + space + "quick"); therefore B04's zero hit is caused
        # by the missing U+1D11E, not by the space in the fixture text.
        assert len(search_anchors(path, "\u1d11quick")) == 1
        assert search_anchors(path, "\U0001d11equick") == []
