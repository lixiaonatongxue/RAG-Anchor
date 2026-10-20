"""Registration of system fonts required by benchmark fixtures.

CJK and non-BMP samples rely on fonts that already exist in the execution
environment.  Font files are never committed to the repository; the dataset is
reproducible only on hosts that provide the same system fonts.

This module must not import ``pypdfium2`` or ``rag_anchor``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


@dataclass(frozen=True, slots=True)
class FontSpec:
    """One system font that benchmark fixtures may reference."""

    name: str
    path: str
    subfont_index: int | None = None


FONT_SPECS: dict[str, FontSpec] = {
    "WQYMicro": FontSpec(
        name="WQYMicro",
        path="/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        subfont_index=0,
    ),
    "FreeSerif": FontSpec(
        name="FreeSerif",
        path="/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    ),
}


def register_benchmark_fonts() -> list[str]:
    """Register every known benchmark font and return unavailable names.

    Registration is idempotent.  A font that is missing on disk or cannot be
    parsed by reportlab is reported instead of silently falling back to a
    ``.notdef`` glyph.
    """
    missing: list[str] = []
    for spec in FONT_SPECS.values():
        try:
            pdfmetrics.getFont(spec.name)
            continue
        except KeyError:
            pass

        font_path = Path(spec.path)
        if not font_path.is_file():
            missing.append(spec.name)
            continue

        try:
            if spec.subfont_index is None:
                font = TTFont(spec.name, str(font_path))
            else:
                font = TTFont(
                    spec.name,
                    str(font_path),
                    subfontIndex=spec.subfont_index,
                )
        except Exception:
            missing.append(spec.name)
            continue

        pdfmetrics.registerFont(font)
    return missing


def ensure_registered(names: Iterable[str]) -> list[str]:
    """Register benchmark fonts and return requested names still unavailable.

    Standard reportlab fonts such as ``Helvetica`` are already registered and
    therefore always report as available.
    """
    register_benchmark_fonts()
    unavailable: list[str] = []
    for name in sorted(set(names)):
        try:
            pdfmetrics.getFont(name)
        except KeyError:
            unavailable.append(name)
    return unavailable
