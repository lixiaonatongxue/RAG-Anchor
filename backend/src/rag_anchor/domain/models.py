"""Framework-independent domain models for PDF evidence anchoring.

Coordinate contract (ADR-0001):

* Units are PDF points (1 pt = 1/72 inch).
* ``PdfRect`` coordinates are absolute PDF user space values, exactly as
  returned by PDFium's ``get_charbox()``. They are never CropBox-relative.
* Origin is the page lower-left corner, x grows right, y grows up.
* Page rotation is *not* baked into a rectangle; it is reported separately in
  :class:`PageDescriptor`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PdfRect:
    """Axis-aligned rectangle in absolute PDF user space points."""

    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        coordinates = (self.left, self.bottom, self.right, self.top)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("PdfRect coordinates must be finite")
        if self.left > self.right or self.bottom > self.top:
            raise ValueError("PdfRect requires left <= right and bottom <= top")

    @property
    def width(self) -> float:
        """Rectangle width in PDF points."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Rectangle height in PDF points."""
        return self.top - self.bottom

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return ``(left, bottom, right, top)``."""
        return (self.left, self.bottom, self.right, self.top)


@dataclass(frozen=True, slots=True)
class PageDescriptor:
    """Page geometry needed to interpret hit rectangles.

    ``cropbox`` is ``(left, bottom, right, top)`` in absolute PDF user space.
    ``width_pt`` / ``height_pt`` are the *unrotated* CropBox dimensions.
    Hit rectangles are also absolute user space, so a non-zero CropBox origin
    is not subtracted from them.
    """

    page_index: int
    width_pt: float
    height_pt: float
    rotation: int
    cropbox: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class AnchorHit:
    """One non-overlapping occurrence of the query in a PDF.

    ``char_start`` / ``char_count`` index into the page text returned by
    PDFium, which joins lines with ``\\r\\n``. ``matched_text`` preserves the
    PDF's original casing and line breaks.

    ``rects`` contains one rectangle per matched text line in absolute PDF
    user space points. It is never a single rectangle spanning multiple
    lines, so the vertical gap between lines is not highlighted.
    """

    page_index: int
    page_number: int
    matched_text: str
    char_start: int
    char_count: int
    rects: tuple[PdfRect, ...]
    page: PageDescriptor

    @property
    def bbox(self) -> PdfRect | None:
        """Smallest rectangle enclosing all line rectangles (derived)."""
        if not self.rects:
            return None
        return PdfRect(
            left=min(rect.left for rect in self.rects),
            bottom=min(rect.bottom for rect in self.rects),
            right=max(rect.right for rect in self.rects),
            top=max(rect.top for rect in self.rects),
        )
