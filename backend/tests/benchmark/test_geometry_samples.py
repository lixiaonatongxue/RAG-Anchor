"""CropBox, rotation, and multi-page geometry regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bench_helpers import DATASET_DIR
from benchmarks import generate, match, runner, truth
from pypdf import PdfReader

from rag_anchor.services import search_anchors

NEW_SAMPLE_IDS = (
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
)
# B03/B04 are the non-BMP boundary/unsupported samples; they must not be
# gated by the direct-quality detection and matching assertions.
BOUNDARY_SAMPLE_IDS = ("B03", "B04")
DIRECT_NEW_SAMPLE_IDS = tuple(
    sample_id for sample_id in NEW_SAMPLE_IDS if sample_id not in BOUNDARY_SAMPLE_IDS
)
ORIGINAL_SAMPLE_IDS = ("A01", "H01", "C01", "B02", "A05")

E01_CROPBOX = [50.0, 100.0, 400.0, 500.0]
E02_CROPBOX = [70.0, 120.0, 450.0, 560.0]
F04_CROPBOX = [50.0, 100.0, 400.0, 500.0]
F04_TRUTH_BOX = [124.012, 197.516, 160.024, 208.616]

E01_TRUTH_BOX = [124.012, 197.516, 160.024, 208.616]
E01_SERVICE_RECT = (124.384, 197.444, 160.012, 208.616)


@pytest.fixture(scope="module")
def specs() -> dict[str, dict[str, Any]]:
    return {spec["id"]: spec for spec in truth.load_specs(DATASET_DIR)}


@pytest.fixture(scope="module")
def pdfs(specs: dict[str, dict[str, Any]], tmp_path_factory) -> dict[str, Path]:
    output = tmp_path_factory.mktemp("geometry-samples")
    return {sample_id: generate.generate_sample(spec, output) for sample_id, spec in specs.items()}


def _hits(sample_id: str, specs: dict[str, dict[str, Any]], pdfs: dict[str, Path]):
    spec = specs[sample_id]
    query = spec["queries"][0]
    return search_anchors(pdfs[sample_id], query["text"], match_case=query["match_case"])


def _truth(sample_id: str, specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return truth.compute_truth(specs[sample_id])


def test_manifest_contains_geometry_samples(specs: dict[str, dict[str, Any]]) -> None:
    assert len(specs) == 18
    assert set(NEW_SAMPLE_IDS) <= set(specs)


@pytest.mark.parametrize("sample_id", NEW_SAMPLE_IDS)
def test_new_sample_spec_is_valid(sample_id: str, specs: dict[str, dict[str, Any]]) -> None:
    assert truth.validate_spec(specs[sample_id]) == []


@pytest.mark.parametrize("sample_id", NEW_SAMPLE_IDS)
def test_new_sample_render_is_deterministic(
    sample_id: str,
    specs: dict[str, dict[str, Any]],
) -> None:
    first = generate.render_pdf_bytes(specs[sample_id])
    second = generate.render_pdf_bytes(specs[sample_id])
    assert first == second


@pytest.mark.parametrize(
    ("sample_id", "cropbox"),
    [("E01", E01_CROPBOX), ("E02", E02_CROPBOX)],
)
def test_cropbox_pdf_metadata(
    sample_id: str,
    cropbox: list[float],
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    assert generate.preflight_pdf(specs[sample_id], pdfs[sample_id]) == []

    page = PdfReader(str(pdfs[sample_id])).pages[0]
    values = [
        float(page.cropbox.left),
        float(page.cropbox.bottom),
        float(page.cropbox.right),
        float(page.cropbox.top),
    ]
    assert values == pytest.approx(cropbox)
    assert int(page.rotation) % 360 == 0
    assert float(page.mediabox.width) == pytest.approx(612.0)
    assert float(page.mediabox.height) == pytest.approx(792.0)


def test_e01_truth_box_uses_absolute_coordinates(
    specs: dict[str, dict[str, Any]],
) -> None:
    occurrence = _truth("E01", specs)["queries"][0]["occurrences"][0]
    assert occurrence["bbox"] == pytest.approx(E01_TRUTH_BOX)
    # A CropBox-relative implementation would return left - 50 and bottom - 100.
    assert occurrence["bbox"][0] != pytest.approx(E01_TRUTH_BOX[0] - 50.0)
    assert occurrence["bbox"][1] != pytest.approx(E01_TRUTH_BOX[1] - 100.0)


def test_e01_service_rect_is_not_cropbox_relative(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hits = _hits("E01", specs, pdfs)
    assert len(hits) == 1

    rect = hits[0].rects[0].as_tuple()
    assert rect[0] == pytest.approx(E01_SERVICE_RECT[0], abs=1.0)
    assert rect[1] == pytest.approx(E01_SERVICE_RECT[1], abs=1.0)
    assert rect[0] != pytest.approx(E01_SERVICE_RECT[0] - 50.0, abs=1.0)
    assert rect[1] != pytest.approx(E01_SERVICE_RECT[1] - 100.0, abs=1.0)

    page = hits[0].page
    assert page.cropbox == pytest.approx(tuple(E01_CROPBOX))
    assert page.width_pt == pytest.approx(350.0)
    assert page.height_pt == pytest.approx(400.0)


def test_e02_truth_keeps_two_absolute_line_boxes(
    specs: dict[str, dict[str, Any]],
) -> None:
    occurrence = _truth("E02", specs)["queries"][0]["occurrences"][0]
    assert [part["text"] for part in occurrence["parts"]] == ["beta", "gamma"]

    beta_box = occurrence["parts"][0]["box"]
    gamma_box = occurrence["parts"][1]["box"]
    assert beta_box == pytest.approx([132.688, 497.516, 156.04, 508.616])
    assert gamma_box == pytest.approx([100.0, 447.516, 140.008, 458.616])
    assert beta_box[1] > gamma_box[3]


def test_e02_service_returns_two_absolute_line_rects(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hits = _hits("E02", specs, pdfs)
    assert len(hits) == 1

    rects = [rect.as_tuple() for rect in hits[0].rects]
    assert len(rects) == 2
    assert rects[0][0] == pytest.approx(132.688, abs=1.0)
    assert rects[1][0] == pytest.approx(100.0, abs=1.0)
    assert rects[0][1] != pytest.approx(497.516 - 120.0, abs=1.0)
    assert rects[1][1] != pytest.approx(447.516 - 120.0, abs=1.0)

    page = hits[0].page
    assert page.cropbox == pytest.approx(tuple(E02_CROPBOX))
    assert page.width_pt == pytest.approx(380.0)
    assert page.height_pt == pytest.approx(440.0)


@pytest.mark.parametrize(
    ("sample_id", "rotation"),
    [("F01", 90), ("F02", 180), ("F03", 270)],
)
def test_rotation_pdf_metadata(
    sample_id: str,
    rotation: int,
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    assert generate.preflight_pdf(specs[sample_id], pdfs[sample_id]) == []

    page = PdfReader(str(pdfs[sample_id])).pages[0]
    assert int(page.rotation) % 360 == rotation
    assert float(page.mediabox.width) == pytest.approx(612.0)
    assert float(page.mediabox.height) == pytest.approx(792.0)


@pytest.mark.parametrize("sample_id", ["F01", "F02", "F03"])
def test_rotation_truth_matches_unrotated_layout(
    sample_id: str,
    specs: dict[str, dict[str, Any]],
) -> None:
    rotated_box = _truth(sample_id, specs)["queries"][0]["occurrences"][0]["bbox"]
    unrotated_box = _truth("A01", specs)["queries"][0]["occurrences"][0]["bbox"]
    assert rotated_box == pytest.approx(unrotated_box)


@pytest.mark.parametrize(
    ("sample_id", "rotation"),
    [("F01", 90), ("F02", 180), ("F03", 270)],
)
def test_rotation_service_returns_unrotated_absolute_coordinates(
    sample_id: str,
    rotation: int,
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hits = _hits(sample_id, specs, pdfs)
    assert len(hits) == 1

    rect = hits[0].rects[0].as_tuple()
    assert rect == pytest.approx(E01_SERVICE_RECT, abs=0.5)

    page = hits[0].page
    assert page.rotation == rotation
    # Rotation must not swap the unrotated CropBox dimensions.
    assert (page.width_pt, page.height_pt) == pytest.approx((612.0, 792.0))


def test_a03_truth_has_one_hit_per_page(specs: dict[str, dict[str, Any]]) -> None:
    sample_truth = _truth("A03", specs)
    occurrences = sample_truth["queries"][0]["occurrences"]

    assert sample_truth["page_count"] == 3
    assert [occurrence["page_index"] for occurrence in occurrences] == [0, 1, 2]
    for occurrence in occurrences:
        assert occurrence["bbox"] == pytest.approx([124.012, 197.516, 166.024, 208.616])


def test_a03_service_reports_page_index_and_page_number(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hits = _hits("A03", specs, pdfs)
    assert len(hits) == 3
    assert [hit.page_index for hit in hits] == [0, 1, 2]
    assert [hit.page_number for hit in hits] == [1, 2, 3]
    for hit in hits:
        assert hit.rects[0].as_tuple() == pytest.approx(
            (124.792, 199.832, 165.22, 208.616),
            abs=0.5,
        )


def test_a03_generated_pdf_has_three_pages(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    assert len(PdfReader(str(pdfs["A03"])).pages) == 3
    assert generate.preflight_pdf(specs["A03"], pdfs["A03"]) == []


@pytest.mark.parametrize("sample_id", NEW_SAMPLE_IDS)
def test_new_sample_truth_has_no_engine_fields(
    sample_id: str,
    specs: dict[str, dict[str, Any]],
) -> None:
    blob = json.dumps(_truth(sample_id, specs), ensure_ascii=False)
    assert "char_start" not in blob
    assert "rects" not in blob


def test_original_samples_detection_does_not_regress(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    for sample_id in ORIGINAL_SAMPLE_IDS:
        expected = _truth(sample_id, specs)["queries"][0]
        hits = _hits(sample_id, specs, pdfs)
        assert len(hits) == expected["expected_count"], sample_id
        assert [hit.page_index for hit in hits] == [
            occurrence["page_index"] for occurrence in expected["occurrences"]
        ], sample_id


@pytest.mark.parametrize("sample_id", DIRECT_NEW_SAMPLE_IDS)
def test_new_sample_matches_without_detection_errors(
    sample_id: str,
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    record = runner.run_sample(specs[sample_id], pdfs[sample_id], repeats=1, warmup=0)
    assert record["engine_error"] is False

    predictions = [query["predicted_hits"] for query in record["queries"]]
    result = match.match_sample(_truth(sample_id, specs), predictions, specs[sample_id])

    for query in result["queries"]:
        assert query["tp"] == query["expected_count"]
        assert query["fp"] == 0
        assert query["fn"] == 0
        assert query["page_counts"]["predicted"] == query["page_counts"]["expected"]
    assert [failure["code"] for failure in result["failures"]] == []


def test_all_eighteen_samples_validate_and_preflight(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    assert set(specs) == set(NEW_SAMPLE_IDS) | set(ORIGINAL_SAMPLE_IDS)
    for sample_id, spec in specs.items():
        assert truth.validate_spec(spec) == [], sample_id
        assert generate.preflight_pdf(spec, pdfs[sample_id]) == [], sample_id


def test_f04_pdf_metadata(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    assert generate.preflight_pdf(specs["F04"], pdfs["F04"]) == []

    page = PdfReader(str(pdfs["F04"])).pages[0]
    assert float(page.mediabox.width) == pytest.approx(612.0)
    assert float(page.mediabox.height) == pytest.approx(792.0)
    cropbox = [
        float(page.cropbox.left),
        float(page.cropbox.bottom),
        float(page.cropbox.right),
        float(page.cropbox.top),
    ]
    assert cropbox == pytest.approx(F04_CROPBOX)
    assert int(page.rotation) % 360 == 90


def test_f04_target_is_inside_visible_cropbox(
    specs: dict[str, dict[str, Any]],
) -> None:
    bbox = _truth("F04", specs)["queries"][0]["occurrences"][0]["bbox"]
    left, bottom, right, top = F04_CROPBOX

    assert left < bbox[0] and bbox[2] < right
    assert bottom < bbox[1] and bbox[3] < top


def test_f04_truth_box_is_unrotated_layout(
    specs: dict[str, dict[str, Any]],
) -> None:
    occurrence = _truth("F04", specs)["queries"][0]["occurrences"][0]

    assert occurrence["parts"][0]["text"] == "zephyr"
    assert occurrence["bbox"] == pytest.approx(F04_TRUTH_BOX)
    assert occurrence["bbox"] == pytest.approx(
        _truth("E01", specs)["queries"][0]["occurrences"][0]["bbox"]
    )


def test_f04_service_reports_expected_hit_and_page(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hits = _hits("F04", specs, pdfs)
    assert len(hits) == 1

    hit = hits[0]
    assert hit.page_index == 0
    assert hit.page_number == 1
    assert hit.matched_text == "zephyr"


def test_f04_service_rect_is_neither_cropbox_relative_nor_rotated(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hits = _hits("F04", specs, pdfs)
    assert len(hits) == 1
    assert len(hits[0].rects) == 1

    rect = hits[0].rects[0].as_tuple()
    # Subtracting the CropBox origin or baking in /Rotate 90 would move the
    # rectangle far outside this tolerance.
    assert rect == pytest.approx(E01_SERVICE_RECT, abs=0.5)
    assert rect[0] != pytest.approx(E01_SERVICE_RECT[0] - 50.0, abs=1.0)
    assert rect[1] != pytest.approx(E01_SERVICE_RECT[1] - 100.0, abs=1.0)


def test_f04_page_descriptor_uses_unrotated_cropbox_size(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    page = _hits("F04", specs, pdfs)[0].page

    assert page.rotation == 90
    assert page.cropbox == pytest.approx(tuple(F04_CROPBOX))
    # 90 degree rotation must not swap the unrotated CropBox dimensions.
    assert page.width_pt == pytest.approx(350.0)
    assert page.height_pt == pytest.approx(400.0)


def test_f04_bbox_and_rects_follow_structural_contract(
    specs: dict[str, dict[str, Any]],
    pdfs: dict[str, Path],
) -> None:
    hit = _hits("F04", specs, pdfs)[0]

    assert len(hit.rects) == 1
    rect = hit.rects[0]
    assert rect.left < rect.right
    assert rect.bottom < rect.top
    assert hit.bbox is not None
    assert hit.bbox.as_tuple() == rect.as_tuple()
