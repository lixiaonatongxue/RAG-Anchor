"""Fixture generation, determinism, and independent preflight checks."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from bench_helpers import DATASET_DIR
from benchmarks import generate, truth
from pypdf import PdfReader


@pytest.fixture(scope="module")
def specs() -> list[dict[str, Any]]:
    return truth.load_specs(DATASET_DIR)


@pytest.fixture(scope="module")
def generated(specs, tmp_path_factory):
    output = tmp_path_factory.mktemp("generated-pdfs")
    paths = generate.generate_dataset(specs, output)
    return list(zip(specs, paths, strict=True))


def test_every_fixture_passes_preflight(generated) -> None:
    assert len(generated) == 5
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
