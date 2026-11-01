"""M1-03B service tests: exact evidence location over the persisted index.

Every test uploads a runtime-generated PDF through the public HTTP API and
builds its index through the same public API. Only afterwards is
``EvidenceLocationService`` called directly, so the tests exercise the real
persisted index rather than a hand-built repository fixture.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from rag_anchor.domain.models import PdfRect
from rag_anchor.services import evidence_service as evidence_module
from rag_anchor.services.evidence_service import (
    EVIDENCE_ITEMS_MAX,
    EvidenceDocumentNotFoundError,
    EvidenceIndexMismatchError,
    EvidenceIndexUnavailableError,
    EvidenceLocationService,
    EvidenceMappingError,
    EvidenceSourceError,
    EvidenceSpan,
    EvidenceValidationError,
    map_python_span,
    verify_mapped_text,
)

DOCUMENTS_URL = "/api/v1/documents"

_DEJAVU_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
)


def _non_bmp_font() -> str:
    """Register a font with supplementary-plane coverage, or skip the test."""
    for candidate in _DEJAVU_CANDIDATES:
        if not candidate.is_file():
            continue
        name = "DejaVuEvidenceProbe"
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(candidate)))
        return name
    pytest.skip("No font with supplementary-plane coverage is available")


_SUPPLEMENTARY_CMAP = b"""/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<00> <FF>
endcodespacerange
1 beginbfchar
<41> <D835DC39>
endbfchar
endcmap
CMapName currentdict /CMap defineresource pop
end
end
"""


def _map_a_to_supplementary_glyph(path: Path) -> Path:
    """Make every ``A`` extract as U+1D439 (a supplementary code point).

    reportlab truncates supplementary characters, so the fixture draws ordinary
    ``A`` glyphs and attaches a ``ToUnicode`` CMap that maps the ``A`` code to
    the UTF-16 surrogate pair. PDFium then exposes a real two-unit
    supplementary character with normal geometry.
    """
    reader = PdfReader(str(path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    stream = DecodedStreamObject()
    stream.set_data(_SUPPLEMENTARY_CMAP)
    for page in writer.pages:
        fonts = page["/Resources"]["/Font"]
        for key in fonts:
            font = fonts[key].get_object()
            if "Helvetica" in str(font.get("/BaseFont")):
                font[NameObject("/ToUnicode")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    path.write_bytes(buffer.getvalue())
    return path


def _upload(client, path: Path) -> UUID:
    response = client.post(
        DOCUMENTS_URL,
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return UUID(response.json()["document_id"])


def _prepare(api, path: Path) -> UUID:
    """Upload and index one PDF through the public HTTP API."""
    document_id = _upload(api.client, path)
    response = api.client.post(f"{DOCUMENTS_URL}/{document_id}/index")
    assert response.status_code == 200, response.text
    return document_id


def _service(api) -> EvidenceLocationService:
    return api.app.state.evidence_service


def _page_text(api, document_id: UUID, page_index: int) -> str:
    repository = _service(api).repository
    pages = repository.get_document_page_texts(document_id)
    return next(page.text for page in pages if page.page_index == page_index)


def _span(
    api,
    document_id: UUID,
    page_index: int,
    needle: str,
    occurrence: int = 0,
    chunk_id: str | None = None,
) -> EvidenceSpan:
    """Build a span for the ``occurrence``-th match of ``needle`` on one page."""
    text = _page_text(api, document_id, page_index)
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(needle, start + 1)
    return EvidenceSpan(
        page_index=page_index,
        match_start=start,
        match_end=start + len(needle),
        hit_text=needle,
        chunk_id=chunk_id,
    )


def _locate_one(
    api,
    document_id: UUID,
    page_index: int,
    needle: str,
    occurrence: int = 0,
):
    span = _span(api, document_id, page_index, needle, occurrence)
    located = _service(api).locate(document_id, [span])
    assert len(located) == 1
    return located[0]


def _long_page(pdf_factory) -> Path:
    """A page of ~1200 characters split into 24 lines, so it has 2+ chunks."""
    lines = [(100.0, 740.0 - 20.0 * index, "a" * 50) for index in range(24)]
    lines[-1] = (100.0, 740.0 - 20.0 * 23, "a" * 44 + "NEEDLE")
    return pdf_factory("long_page.pdf", [lines])


def test_bmp_span_returns_real_geometry(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)

    item = _locate_one(api, document_id, 0, "fox")

    assert item.document_id == document_id
    assert item.page_index == 0
    assert item.page_number == 1
    assert item.hit_text == "fox"
    assert item.matched_text == "fox"
    assert item.chunk_id is None
    assert item.char_start == item.match_start
    assert item.char_count == 3
    assert len(item.rects) == 1
    assert all(isinstance(rect, PdfRect) for rect in item.rects)
    assert item.bbox == item.rects[0]
    assert 0.0 < item.bbox.left < item.bbox.right < item.page.width_pt
    assert 0.0 < item.bbox.bottom < item.bbox.top < item.page.height_pt
    assert item.page.page_index == 0
    assert item.page.width_pt == pytest.approx(612.0)
    assert item.page.height_pt == pytest.approx(792.0)
    assert item.page.rotation == 0
    assert item.page.cropbox == pytest.approx((0.0, 0.0, 612.0, 792.0))


def test_non_bmp_prefix_maps_to_a_later_pdfium_index(api, non_bmp_pdf: Path) -> None:
    document_id = _prepare(api, non_bmp_pdf)

    item = _locate_one(api, document_id, 0, "quick")

    assert item.match_start == 0
    assert item.char_start == 1
    assert item.char_start != item.match_start
    assert item.char_count == 5
    assert item.matched_text == "quick"
    assert item.rects


def test_consecutive_non_bmp_prefixes_are_skipped(api, pdf_factory) -> None:
    path = pdf_factory(
        "two_non_bmp.pdf",
        [[(100.0, 700.0, "\U0001d11e\U0001d11equick")]],
        font_name=_non_bmp_font(),
    )
    document_id = _prepare(api, path)

    item = _locate_one(api, document_id, 0, "quick")

    assert item.match_start == 0
    assert item.char_start == 2
    assert item.char_count == 5
    assert item.matched_text == "quick"


def test_non_bmp_immediately_before_the_span_is_skipped(api, pdf_factory) -> None:
    path = pdf_factory(
        "inner_non_bmp.pdf",
        [[(100.0, 700.0, "a\U0001d11eb quick")]],
        font_name=_non_bmp_font(),
    )
    document_id = _prepare(api, path)

    item = _locate_one(api, document_id, 0, "b quick")

    assert item.match_start == 1
    assert item.char_start == 2
    assert item.char_count == 7
    assert item.matched_text == "b quick"
    assert item.rects


def test_non_bmp_inside_the_span_is_rejected(api, pdf_factory) -> None:
    """An unrepresentable character inside the span has no exact geometry."""
    path = pdf_factory(
        "inner_non_bmp_span.pdf",
        [[(100.0, 700.0, "a\U0001d11eb quick")]],
        font_name=_non_bmp_font(),
    )
    document_id = _prepare(api, path)
    text = _page_text(api, document_id, 0)
    assert text == "ab quick"

    # PDFium keeps one internal character for U+1D11E, returns no Python text
    # for it, and still reports a real character box. Mapping "ab" would
    # include that glyph, so the range must be rejected instead of returning
    # geometry the caller never asked for.
    span = EvidenceSpan(0, 0, 2, "ab")
    with pytest.raises(EvidenceMappingError, match="produces no text"):
        _service(api).locate(document_id, [span])


def test_supplementary_character_is_located(api, pdf_factory) -> None:
    path = _map_a_to_supplementary_glyph(
        pdf_factory("supplementary.pdf", [[(100.0, 700.0, "A quick fox")]])
    )
    document_id = _prepare(api, path)
    assert _page_text(api, document_id, 0) == "\U0001d439 quick fox"

    item = _locate_one(api, document_id, 0, "quick")

    assert item.match_start == 2
    assert item.char_start == 3
    assert item.char_count == 5
    assert item.matched_text == "quick"
    assert item.rects


def test_supplementary_character_itself_is_located(api, pdf_factory) -> None:
    path = _map_a_to_supplementary_glyph(
        pdf_factory("supplementary_span.pdf", [[(100.0, 700.0, "A quick fox")]])
    )
    document_id = _prepare(api, path)

    item = _locate_one(api, document_id, 0, "\U0001d439")

    assert item.match_start == 0
    assert item.match_end == 1
    assert item.char_start == 0
    assert item.char_count == 2
    assert item.matched_text == "\U0001d439"
    assert item.rects


def test_span_ending_after_supplementary_character_keeps_both_units(api, pdf_factory) -> None:
    path = _map_a_to_supplementary_glyph(
        pdf_factory("supplementary_trailing.pdf", [[(100.0, 700.0, "A quick fox")]])
    )
    document_id = _prepare(api, path)

    item = _locate_one(api, document_id, 0, "\U0001d439 ")

    assert item.match_start == 0
    assert item.match_end == 2
    assert item.char_start == 0
    assert item.char_count == 3
    assert item.matched_text == "\U0001d439 "
    assert item.rects


def test_legacy_supplementary_prefix_seed_locates(api, pdf_factory) -> None:
    """Reproduce the E2-Q1 shape: supplementary prefixes shift PDFium indices."""
    path = _map_a_to_supplementary_glyph(
        pdf_factory(
            "legacy_supplementary_seed.pdf",
            [[(100.0, 700.0, "AAAAA quick fox")]],
        )
    )
    document_id = _prepare(api, path)
    text = _page_text(api, document_id, 0)
    assert text == "\U0001d439\U0001d439\U0001d439\U0001d439\U0001d439 quick fox"
    assert text.index("quick") == 6

    item = _locate_one(api, document_id, 0, "quick")

    assert item.match_start == 6
    assert item.char_start == 11
    assert item.char_count == 5
    assert item.matched_text == "quick"
    assert item.rects


def test_span_with_spaces_skips_zero_area_boxes(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)

    item = _locate_one(api, document_id, 0, "quick brown")

    assert item.rects
    assert all(rect.right > rect.left and rect.top > rect.bottom for rect in item.rects)


def test_continuous_range_mismatch_is_rejected() -> None:
    class RewritingTextPage:
        """Stand-in for PDFium rewriting text when a range is requested."""

        def get_text_range(self, start: int, count: int) -> str:
            return "ab?"

    with pytest.raises(EvidenceMappingError, match="continuous character range"):
        verify_mapped_text(RewritingTextPage(), 0, 2, "ab")


def test_map_python_span_rejects_a_boundary_inside_one_unit() -> None:
    # Two PDFium characters, each expressed as two Python characters.
    offsets = [0, 2, 4]

    assert map_python_span(offsets, 0, 2) == (0, 1)
    assert map_python_span(offsets, 2, 4) == (1, 2)
    with pytest.raises(EvidenceMappingError, match="match_start falls inside"):
        map_python_span(offsets, 1, 2)
    with pytest.raises(EvidenceMappingError, match="match_end falls inside"):
        map_python_span(offsets, 0, 1)
    with pytest.raises(EvidenceMappingError, match="empty offset map"):
        map_python_span([], 0, 1)


def test_map_python_span_handles_supplementary_continuation_units() -> None:
    # U+1D439 then " quick" as units: high, low (continuation), space, q, u, i, c, k.
    offsets = [0, 1, 1, 2, 3, 4, 5, 6, 7]
    continuations = [False, True, False, False, False, False, False, False]

    assert map_python_span(offsets, 0, 1, continuations) == (0, 2)
    assert map_python_span(offsets, 1, 2, continuations) == (2, 3)
    assert map_python_span(offsets, 2, 7, continuations) == (3, 8)


def test_map_python_span_validates_the_requested_range() -> None:
    offsets = [0, 1, 2]

    with pytest.raises(EvidenceValidationError, match="match_start must be zero"):
        map_python_span(offsets, -1, 1)
    with pytest.raises(EvidenceValidationError, match="match_end must be greater"):
        map_python_span(offsets, 1, 1)
    with pytest.raises(EvidenceValidationError, match="span is outside"):
        map_python_span(offsets, 0, 3)


def test_repeated_occurrences_are_located_by_offsets(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    spans = [_span(api, document_id, 0, "quick", index) for index in range(3)]

    located = _service(api).locate(document_id, spans)

    assert [item.match_start for item in located] == [span.match_start for span in spans]
    assert [item.hit_text for item in located] == ["quick", "quick", "quick"]
    assert [item.matched_text for item in located] == ["quick", "quick", "quick"]
    assert all(item.char_count == 5 for item in located)
    rectangles = [item.rects[0].as_tuple() for item in located]
    assert len(set(rectangles)) == 3


def test_multi_page_identical_text_uses_the_requested_page(api, multi_page_pdf: Path) -> None:
    document_id = _prepare(api, multi_page_pdf)
    spans = [_span(api, document_id, index, "quick") for index in range(3)]

    located = _service(api).locate(document_id, spans)

    assert [item.page_index for item in located] == [0, 1, 2]
    assert [item.page_number for item in located] == [1, 2, 3]
    assert all(item.hit_text == "quick" for item in located)
    assert all(item.rects for item in located)


def test_crlf_span_returns_one_rectangle_per_line(api, pdf_factory) -> None:
    path = pdf_factory(
        "crlf.pdf",
        [[(100.0, 700.0, "alpha quick"), (100.0, 650.0, "brown beta")]],
    )
    document_id = _prepare(api, path)
    text = _page_text(api, document_id, 0)
    assert "quick\r\nbrown" in text

    item = _locate_one(api, document_id, 0, "quick\r\nbrown")

    assert item.match_start == 6
    assert item.match_end == 18
    assert item.char_count == 12
    assert len(item.rects) == 2
    assert item.rects[0].bottom > item.rects[1].bottom
    assert item.bbox is not None
    assert item.bbox.left == min(rect.left for rect in item.rects)
    assert item.bbox.bottom == min(rect.bottom for rect in item.rects)
    assert item.bbox.right == max(rect.right for rect in item.rects)
    assert item.bbox.top == max(rect.top for rect in item.rects)


def test_cropbox_origin_is_not_subtracted(api, cropbox_pdf: Path) -> None:
    document_id = _prepare(api, cropbox_pdf)

    item = _locate_one(api, document_id, 0, "fox")

    assert item.page.cropbox == pytest.approx((50.0, 100.0, 400.0, 500.0))
    assert item.page.width_pt == pytest.approx(350.0)
    assert item.page.height_pt == pytest.approx(400.0)
    # The text was drawn at x=100 in absolute user space.
    assert item.bbox.left > 90.0
    assert item.bbox.bottom > 190.0


def test_rotation_is_reported_not_baked_into_rectangles(api, pdf_factory) -> None:
    path = pdf_factory(
        "rotated.pdf",
        [[(100.0, 200.0, "The quick brown fox")]],
        rotation=90,
    )
    document_id = _prepare(api, path)
    text = _page_text(api, document_id, 0)
    start = text.index("fox")

    item = _locate_one(api, document_id, 0, "fox")

    assert start >= 0
    assert item.page.rotation == 90
    assert item.page.width_pt == pytest.approx(612.0)
    assert item.page.height_pt == pytest.approx(792.0)
    assert item.rects
    assert all(rect.width > 0.0 and rect.height > 0.0 for rect in item.rects)


def test_matching_chunk_id_is_verified_and_echoed(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    span = _span(api, document_id, 0, "fox")
    chunks = _service(api).repository.get_document_chunks(document_id)
    chunk = next(
        item
        for item in chunks
        if item.start_offset <= span.match_start and span.match_end <= item.end_offset
    )
    span = dataclasses.replace(span, chunk_id=chunk.chunk_id)

    located = _service(api).locate(document_id, [span])

    assert located[0].chunk_id == chunk.chunk_id
    assert located[0].matched_text == "fox"


def test_unknown_chunk_id_is_rejected(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    span = _span(api, document_id, 0, "fox", chunk_id="chk_does_not_exist")

    with pytest.raises(EvidenceValidationError, match="chunk_id does not exist"):
        _service(api).locate(document_id, [span])


def test_chunk_from_another_document_is_rejected(
    api, single_page_pdf: Path, multi_page_pdf: Path
) -> None:
    first = _prepare(api, single_page_pdf)
    second = _prepare(api, multi_page_pdf)
    foreign_chunk = _service(api).repository.get_document_chunks(second)[0]
    span = _span(api, first, 0, "fox", chunk_id=foreign_chunk.chunk_id)

    with pytest.raises(
        EvidenceValidationError,
        match="chunk_id does not belong to this document",
    ):
        _service(api).locate(first, [span])


def test_chunk_from_another_page_is_rejected(api, multi_page_pdf: Path) -> None:
    document_id = _prepare(api, multi_page_pdf)
    first_page_chunk = _service(api).repository.get_document_chunks(document_id)[0]
    assert first_page_chunk.page_index == 0
    span = _span(api, document_id, 1, "quick", chunk_id=first_page_chunk.chunk_id)

    with pytest.raises(EvidenceValidationError, match="chunk_id does not belong to this page"):
        _service(api).locate(document_id, [span])


def test_span_outside_the_referenced_chunk_is_rejected(api, pdf_factory) -> None:
    document_id = _prepare(api, _long_page(pdf_factory))
    chunks = _service(api).repository.get_document_chunks(document_id)
    assert len(chunks) >= 2
    span = _span(api, document_id, 0, "NEEDLE", chunk_id=chunks[0].chunk_id)

    with pytest.raises(EvidenceValidationError, match="span is outside the referenced chunk"):
        _service(api).locate(document_id, [span])


@pytest.mark.parametrize(
    ("span", "message"),
    [
        (EvidenceSpan(5, 0, 1, "T"), "page_index 5 is outside the indexed document"),
        (EvidenceSpan(True, 0, 1, "T"), "page_index must be an integer"),
        (EvidenceSpan(0, -1, 3, "The"), "span must satisfy 0 <= match_start < match_end"),
        (EvidenceSpan(0, 5, 5, ""), "span must satisfy 0 <= match_start < match_end"),
        (EvidenceSpan(0, 0, 10_000, "x"), "span is outside the stored page text"),
        (EvidenceSpan(0, 0, 3, "Xhe"), "hit_text does not match the stored page text"),
        (EvidenceSpan(0, 0.0, 1, "T"), "match_start must be an integer"),
        (EvidenceSpan(0, 0, 1, 5), "hit_text must be a string"),
    ],
)
def test_invalid_spans_are_rejected(
    api,
    single_page_pdf: Path,
    span: EvidenceSpan,
    message: str,
) -> None:
    document_id = _prepare(api, single_page_pdf)

    with pytest.raises(EvidenceValidationError, match=message):
        _service(api).locate(document_id, [span])


def test_batch_size_is_bounded(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    service = _service(api)
    span = EvidenceSpan(0, 0, 1, "T")

    with pytest.raises(EvidenceValidationError, match="items must contain between"):
        service.locate(document_id, [])
    with pytest.raises(EvidenceValidationError, match="items must contain between"):
        service.locate(document_id, [span] * (EVIDENCE_ITEMS_MAX + 1))


def test_batch_is_all_or_nothing(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    service = _service(api)
    valid = _span(api, document_id, 0, "fox")
    invalid = EvidenceSpan(0, 0, 3, "Xhe")

    with pytest.raises(EvidenceValidationError, match="hit_text does not match"):
        service.locate(document_id, [valid, invalid])

    # The same valid span still works on its own: no partial state was kept.
    assert service.locate(document_id, [valid])[0].matched_text == "fox"


def test_batch_preserves_order_and_duplicates(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    second = _span(api, document_id, 0, "quick", 1)
    first = _span(api, document_id, 0, "fox")

    located = _service(api).locate(document_id, [second, first, second])

    assert [item.match_start for item in located] == [
        second.match_start,
        first.match_start,
        second.match_start,
    ]
    assert located[0].rects == located[2].rects
    assert located[1].hit_text == "fox"


def test_unknown_document_is_rejected(api) -> None:
    with pytest.raises(EvidenceDocumentNotFoundError, match="Document not found"):
        _service(api).locate(uuid4(), [EvidenceSpan(0, 0, 1, "T")])


def test_unindexed_document_is_rejected(api, single_page_pdf: Path) -> None:
    document_id = _upload(api.client, single_page_pdf)

    with pytest.raises(EvidenceIndexUnavailableError, match="not been indexed"):
        _service(api).locate(document_id, [EvidenceSpan(0, 0, 1, "T")])


def test_tampered_pdf_is_rejected(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    stored_pdf = api.settings.documents_dir / f"{document_id}.pdf"
    stored_pdf.write_bytes(stored_pdf.read_bytes() + b"\n% tampered after indexing\n")

    with pytest.raises(EvidenceIndexMismatchError, match="does not match the index"):
        _service(api).locate(document_id, [_span(api, document_id, 0, "fox")])


def test_missing_pdf_is_rejected(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    span = _span(api, document_id, 0, "fox")
    stored_pdf = api.settings.documents_dir / f"{document_id}.pdf"
    stored_pdf.unlink()

    with pytest.raises(EvidenceSourceError, match="missing or unreadable"):
        _service(api).locate(document_id, [span])


def test_stored_page_text_mismatch_is_rejected(api, single_page_pdf: Path, monkeypatch) -> None:
    document_id = _prepare(api, single_page_pdf)
    repository = _service(api).repository
    original_pages = repository.get_document_page_texts(document_id)
    altered_pages = [
        dataclasses.replace(page, text=page.text.replace("lazy", "LAZY")) for page in original_pages
    ]
    assert altered_pages[0].text != original_pages[0].text
    span = _span(api, document_id, 0, "fox")

    monkeypatch.setattr(repository, "get_document_page_texts", lambda _: altered_pages)

    with pytest.raises(EvidenceIndexMismatchError, match="Current page text does not match"):
        _service(api).locate(document_id, [span])


def test_each_request_opens_the_pdf_once_and_maps_a_page_once(
    api, single_page_pdf: Path, monkeypatch
) -> None:
    document_id = _prepare(api, single_page_pdf)
    spans = [_span(api, document_id, 0, "quick", index) for index in range(3)]
    original_document = evidence_module.PdfDocument
    original_map = evidence_module.build_offset_map
    opens: list[str] = []
    maps: list[object] = []

    def counting_document(*args, **kwargs):
        opens.append(args[0])
        return original_document(*args, **kwargs)

    def counting_map(text_page):
        maps.append(text_page)
        return original_map(text_page)

    monkeypatch.setattr(evidence_module, "PdfDocument", counting_document)
    monkeypatch.setattr(evidence_module, "build_offset_map", counting_map)

    located = _service(api).locate(document_id, spans)

    assert len(located) == 3
    assert len(opens) == 1
    assert len(maps) == 1


def test_each_involved_page_is_mapped_once(api, multi_page_pdf: Path, monkeypatch) -> None:
    document_id = _prepare(api, multi_page_pdf)
    spans = [_span(api, document_id, index, "quick") for index in range(3)]
    original_document = evidence_module.PdfDocument
    original_map = evidence_module.build_offset_map
    opens: list[str] = []
    maps: list[object] = []

    def counting_document(*args, **kwargs):
        opens.append(args[0])
        return original_document(*args, **kwargs)

    def counting_map(text_page):
        maps.append(text_page)
        return original_map(text_page)

    monkeypatch.setattr(evidence_module, "PdfDocument", counting_document)
    monkeypatch.setattr(evidence_module, "build_offset_map", counting_map)

    located = _service(api).locate(document_id, spans)

    assert len(located) == 3
    assert len(opens) == 1
    assert len(maps) == 3


def test_locate_is_read_only(api, single_page_pdf: Path) -> None:
    document_id = _prepare(api, single_page_pdf)
    service = _service(api)
    repository = service.repository
    pages_before = repository.get_document_page_texts(document_id)
    chunks_before = repository.get_document_chunks(document_id)
    metadata_before = repository.get_document_index_metadata(document_id)
    database_before = api.settings.database_path.read_bytes()

    service.locate(document_id, [_span(api, document_id, 0, "fox")])

    assert repository.get_document_page_texts(document_id) == pages_before
    assert repository.get_document_chunks(document_id) == chunks_before
    assert repository.get_document_index_metadata(document_id) == metadata_before
    assert api.settings.database_path.read_bytes() == database_before


def test_trailing_non_bmp_after_the_span_is_excluded(api, pdf_factory) -> None:
    path = pdf_factory(
        "trailing_non_bmp.pdf",
        [[(100.0, 700.0, "quick\U0001d11e")]],
        font_name=_non_bmp_font(),
    )
    document_id = _prepare(api, path)

    item = _locate_one(api, document_id, 0, "quick")

    assert item.match_start == 0
    assert item.match_end == 5
    assert item.char_start == 0
    assert item.char_count == 5
    assert item.matched_text == "quick"
    assert item.rects


def test_bmp_cjk_span_keeps_python_offsets(api, cjk_pdf: Path) -> None:
    document_id = _prepare(api, cjk_pdf)

    item = _locate_one(api, document_id, 0, "quick")

    assert item.char_start == item.match_start
    assert item.char_count == 5
    assert item.matched_text == "quick"
    assert item.rects


def test_boundary_inside_a_multi_character_unit_fails_explicitly(
    api, single_page_pdf: Path, monkeypatch
) -> None:
    """A PDFium unit that yields two Python characters has no exact boundary."""
    document_id = _prepare(api, single_page_pdf)
    text = _page_text(api, document_id, 0)
    merge_at = text.index("quick")
    original_map = evidence_module.build_offset_map

    def merged_map(text_page):
        segments, _, _ = original_map(text_page)
        segments[merge_at] = segments[merge_at] + segments[merge_at + 1]
        segments[merge_at + 1] = ""
        offsets = [0]
        for segment in segments:
            offsets.append(offsets[-1] + len(segment))
        return segments, offsets, [False] * len(segments)

    monkeypatch.setattr(evidence_module, "build_offset_map", merged_map)
    span = EvidenceSpan(0, merge_at, merge_at + 1, text[merge_at])

    with pytest.raises(EvidenceMappingError, match="falls inside a character"):
        _service(api).locate(document_id, [span])


def test_corrupt_pdf_is_rejected(api, single_page_pdf: Path, monkeypatch) -> None:
    document_id = _prepare(api, single_page_pdf)
    metadata = _service(api).repository.get_document_index_metadata(document_id)
    assert metadata is not None
    stored_pdf = api.settings.documents_dir / f"{document_id}.pdf"
    stored_pdf.write_bytes(b"%PDF-1.7\nthis is not a usable pdf\n")
    # Bypass the byte-identity check so the PDFium open/parse failure is reached.
    monkeypatch.setattr(evidence_module, "hash_file", lambda _: metadata.pdf_sha256)

    with pytest.raises(EvidenceSourceError, match="could not be"):
        _service(api).locate(document_id, [_span(api, document_id, 0, "fox")])
