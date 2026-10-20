"""Tests for traceable PDF text extraction and deterministic chunking.

PDFs are generated at runtime by the shared unit fixtures; no binary document is
stored in the repository. Chunk offsets are Python ``str`` slice boundaries into
the extracted page text, never PDFium character indices or PDF coordinates.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from rag_anchor.domain import (
    TEXT_EXTRACTION_VERSION,
    PageText,
    TextChunk,
    build_chunk_id,
)
from rag_anchor.services import (
    DEFAULT_MAX_CHARS,
    PdfTextExtractionError,
    PdfTextOpenError,
    chunk_page_texts,
    extract_page_texts,
    normalize_document_id,
    search_anchors,
)

DOC_ID = "11111111-1111-4111-8111-111111111111"


def _page(index: int = 0, text: str = "abcdefghij") -> PageText:
    return PageText(
        document_id=DOC_ID,
        page_index=index,
        page_number=index + 1,
        text=text,
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_single_page_preserves_pdfium_line_breaks(single_page_pdf: Path) -> None:
    pages = extract_page_texts(single_page_pdf, document_id=DOC_ID)

    assert len(pages) == 1
    page = pages[0]
    assert page.document_id == DOC_ID
    assert page.page_index == 0
    assert page.page_number == 1
    assert page.text.splitlines() == [
        "The quick brown fox jumps over the lazy dog.",
        "quick quick QUICK",
    ]
    assert "\r\n" in page.text
    assert page.text.count("quick") == 3
    assert page.text.startswith("The quick")
    assert page.text.endswith("QUICK")


def test_extract_multi_page_never_concatenates_pages(multi_page_pdf: Path) -> None:
    pages = extract_page_texts(multi_page_pdf, document_id=DOC_ID)

    assert [page.page_index for page in pages] == [0, 1, 2]
    assert [page.page_number for page in pages] == [1, 2, 3]
    for expected_page, page in enumerate(pages, start=1):
        assert page.text.splitlines() == [
            f"Page {expected_page} alpha beta gamma",
            "second line quick",
        ]
    assert "Page 2" not in pages[0].text
    assert "Page 3" not in pages[1].text


def test_extract_empty_page_returns_empty_text_and_no_chunks(pdf_factory) -> None:
    blank_pdf = pdf_factory("blank.pdf", [[]])

    pages = extract_page_texts(blank_pdf, document_id=DOC_ID)

    assert len(pages) == 1
    assert pages[0].text == ""
    assert chunk_page_texts(pages, max_chars=10) == []


def test_extract_cjk_text_uses_python_string_offsets(cjk_pdf: Path) -> None:
    pages = extract_page_texts(cjk_pdf, document_id=DOC_ID)

    page = pages[0]
    assert page.text == "中文quick证据"
    assert len(page.text) == 9
    assert page.text.find("quick") == 2
    assert page.text[2:7] == "quick"

    chunks = chunk_page_texts(pages, max_chars=4)

    assert [chunk.text for chunk in chunks] == ["中文qu", "ick证", "据"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 9),
    ]
    assert "".join(chunk.text for chunk in chunks) == page.text


def test_missing_file_and_directory_raise_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing\.pdf"):
        extract_page_texts(tmp_path / "missing.pdf", document_id=DOC_ID)
    with pytest.raises(IsADirectoryError):
        extract_page_texts(tmp_path, document_id=DOC_ID)


def test_corrupted_pdf_raises_clear_extraction_error(tmp_path: Path) -> None:
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"%PDF-1.4\nthis is not a valid pdf\n%%EOF")

    with pytest.raises(PdfTextOpenError, match="Failed to open PDF"):
        extract_page_texts(broken_pdf, document_id=DOC_ID)

    assert issubclass(PdfTextOpenError, PdfTextExtractionError)


# ---------------------------------------------------------------------------
# Chunk boundaries
# ---------------------------------------------------------------------------


def test_chunk_respects_max_chars_and_keeps_short_final_segment() -> None:
    page = _page(text="abcdefghij")

    chunks = chunk_page_texts([page], max_chars=4)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert [chunk.start_offset for chunk in chunks] == [0, 4, 8]
    assert [chunk.end_offset for chunk in chunks] == [4, 8, 10]
    assert all(len(chunk.text) <= 4 for chunk in chunks)
    assert len(chunks[-1].text) < 4
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_chunk_overlap_is_exact_and_traceable() -> None:
    page = _page(text="abcdefghij")

    chunks = chunk_page_texts([page], max_chars=5, overlap_chars=2)

    assert [chunk.text for chunk in chunks] == ["abcde", "defgh", "ghij"]
    assert [(chunk.start_offset, chunk.end_offset) for chunk in chunks] == [
        (0, 5),
        (3, 8),
        (6, 10),
    ]
    assert chunks[0].text[-2:] == chunks[1].text[:2] == "de"
    assert chunks[1].text[-2:] == chunks[2].text[:2] == "gh"
    for chunk in chunks:
        chunk.verify_source(page)
        assert page.text[chunk.start_offset : chunk.end_offset] == chunk.text


def test_zero_overlap_reconstructs_text_without_normalization() -> None:
    text = "Line 1.\r\nLine 2, 中文!"
    page = _page(text=text)

    chunks = chunk_page_texts([page], max_chars=7, overlap_chars=0)

    assert "".join(chunk.text for chunk in chunks) == text
    assert any("\r\n" in chunk.text for chunk in chunks)
    assert any("中文" in chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= 7 for chunk in chunks)
    assert all(chunk.end_offset - chunk.start_offset == len(chunk.text) for chunk in chunks)


def test_chunks_never_span_pages_and_keep_page_numbers(multi_page_pdf: Path) -> None:
    pages = extract_page_texts(multi_page_pdf, document_id=DOC_ID)
    chunks = chunk_page_texts(pages, max_chars=10, overlap_chars=0)
    pages_by_index = {page.page_index: page for page in pages}

    assert {chunk.page_index for chunk in chunks} == {0, 1, 2}
    for page_index in (0, 1, 2):
        page_chunks = [chunk for chunk in chunks if chunk.page_index == page_index]
        assert page_chunks
        assert all(chunk.page_number == page_index + 1 for chunk in page_chunks)
        assert all(chunk.document_id == DOC_ID for chunk in page_chunks)
        assert "".join(chunk.text for chunk in page_chunks) == pages_by_index[page_index].text
        for chunk in page_chunks:
            chunk.verify_source(pages_by_index[page_index])

    page_zero_text = [chunk.text for chunk in chunks if chunk.page_index == 0]
    assert all("Page 2" not in text and "Page 3" not in text for text in page_zero_text)


def test_whitespace_only_page_is_preserved_verbatim() -> None:
    text = "\r\n  \r\n"
    page = _page(text=text)

    chunks = chunk_page_texts([page], max_chars=2)

    assert [chunk.text for chunk in chunks] == ["\r\n", "  ", "\r\n"]
    assert "".join(chunk.text for chunk in chunks) == text


def test_default_chunk_length_is_800_characters() -> None:
    page = _page(text="x" * (DEFAULT_MAX_CHARS + 1))

    chunks = chunk_page_texts([page])

    assert [len(chunk.text) for chunk in chunks] == [DEFAULT_MAX_CHARS, 1]
    assert chunks[0].extraction_version == TEXT_EXTRACTION_VERSION


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_chunk_parameters_raise_value_error(max_chars: int, overlap_chars: int) -> None:
    with pytest.raises(ValueError):
        chunk_page_texts([_page()], max_chars=max_chars, overlap_chars=overlap_chars)


def test_non_integer_chunk_parameters_raise_type_error() -> None:
    with pytest.raises(TypeError, match="max_chars"):
        chunk_page_texts([_page()], max_chars="5")
    with pytest.raises(TypeError, match="overlap_chars"):
        chunk_page_texts([_page()], max_chars=5, overlap_chars=True)


def test_chunk_rejects_mixed_documents_and_duplicate_pages() -> None:
    page_a = PageText(document_id="doc-a", page_index=0, page_number=1, text="a")
    page_b = PageText(document_id="doc-b", page_index=1, page_number=2, text="b")
    duplicate_page = PageText(document_id="doc-a", page_index=0, page_number=1, text="b")

    with pytest.raises(ValueError, match="same document_id"):
        chunk_page_texts([page_a, page_b])
    with pytest.raises(ValueError, match="duplicate page_index"):
        chunk_page_texts([page_a, duplicate_page])


def test_page_and_chunk_models_reject_inconsistent_page_numbers() -> None:
    with pytest.raises(ValueError, match="page_number"):
        PageText(document_id=DOC_ID, page_index=0, page_number=2, text="x")
    with pytest.raises(ValueError, match="page_number"):
        TextChunk(
            chunk_id="chk_x",
            document_id=DOC_ID,
            page_index=0,
            page_number=2,
            start_offset=0,
            end_offset=1,
            text="x",
        )


def test_verify_source_rejects_same_length_but_wrong_text() -> None:
    page = _page(text="abcdef")
    chunk = chunk_page_texts([page], max_chars=3)[0]
    chunk.verify_source(page)

    tampered = TextChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        page_index=chunk.page_index,
        page_number=chunk.page_number,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        text="XYZ",
        chunk_index=chunk.chunk_index,
        extraction_version=chunk.extraction_version,
    )

    assert len(tampered.text) == len(chunk.text)
    with pytest.raises(ValueError, match="exact slice"):
        tampered.verify_source(page)


# ---------------------------------------------------------------------------
# Identity and document id
# ---------------------------------------------------------------------------


def test_uuid_document_id_is_normalized_to_canonical_string(
    single_page_pdf: Path,
) -> None:
    document_id = uuid4()

    pages = extract_page_texts(single_page_pdf, document_id=document_id)

    assert all(page.document_id == str(document_id) for page in pages)
    chunks = chunk_page_texts(pages, max_chars=32)
    assert all(chunk.document_id == str(document_id) for chunk in chunks)
    assert normalize_document_id(document_id) == str(document_id)
    assert normalize_document_id(f"  {DOC_ID}  ") == DOC_ID
    with pytest.raises(TypeError, match="UUID or str"):
        normalize_document_id(123)
    with pytest.raises(ValueError, match="non-empty"):
        normalize_document_id("   ")


def test_chunk_id_is_deterministic_and_built_from_the_same_inputs() -> None:
    pages = [_page(text="abcdefghij")]

    first = chunk_page_texts(pages, max_chars=4)
    second = chunk_page_texts(pages, max_chars=4)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    for chunk in first:
        assert chunk.chunk_id.startswith("chk_")
        assert len(chunk.chunk_id) == 36
        assert chunk.chunk_id == build_chunk_id(
            document_id=DOC_ID,
            page_index=0,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
        )
    assert len({chunk.chunk_id for chunk in first}) == len(first)


def test_chunk_id_is_stable_across_python_hash_seeds() -> None:
    code = (
        "from rag_anchor.domain.text import build_chunk_id;"
        "print(build_chunk_id(document_id='doc', page_index=0,"
        " start_offset=0, end_offset=5))"
    )
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        outputs.append(result.stdout.strip())

    assert outputs[0] == outputs[1]
    assert outputs[0].startswith("chk_")


# ---------------------------------------------------------------------------
# Regression guard
# ---------------------------------------------------------------------------


def test_existing_exact_search_still_works(single_page_pdf: Path) -> None:
    hits = search_anchors(single_page_pdf, "fox")

    assert len(hits) == 1
    assert hits[0].matched_text == "fox"
    assert hits[0].page_index == 0
    assert hits[0].page_number == 1
