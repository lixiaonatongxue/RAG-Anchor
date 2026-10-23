import type { AnchorHit, LocatedEvidence } from "../api/types";

/**
 * The hits currently drawn on the overlay plus the active hit inside that
 * list. Anchor search and chunk evidence both feed this single structure, so
 * there is exactly one rendering path and one coordinate conversion.
 */
export interface HighlightSelection {
  hits: AnchorHit[];
  activeIndex: number;
}

export function emptySelection(): HighlightSelection {
  return { hits: [], activeIndex: -1 };
}

function normalizeActiveIndex(index: number, length: number): number {
  return index >= 0 && index < length ? index : -1;
}

/** Selection used by the existing anchor search result list. */
export function anchorSelection(
  hits: readonly AnchorHit[],
  activeIndex: number,
): HighlightSelection {
  return {
    hits: [...hits],
    activeIndex: normalizeActiveIndex(activeIndex, hits.length),
  };
}

/**
 * Convert one located evidence span into the renderer's hit shape.
 *
 * Only server-provided values are copied. The rectangles stay in absolute PDF
 * user space (ADR-0001); the renderer applies the viewport conversion.
 */
export function locatedEvidenceToHit(item: LocatedEvidence): AnchorHit {
  return {
    page_index: item.page_index,
    page_number: item.page_number,
    matched_text: item.matched_text,
    char_start: item.char_start,
    char_count: item.char_count,
    rects: item.rects,
    bbox: item.bbox,
    page: item.page,
  };
}

/** Selection used after evidence/locate returned exactly one span. */
export function evidenceSelection(item: LocatedEvidence | null): HighlightSelection {
  if (!item) {
    return emptySelection();
  }
  return { hits: [locatedEvidenceToHit(item)], activeIndex: 0 };
}

/**
 * Pick the hits that belong to the rendered page and locate the active one.
 *
 * Switching pages therefore cannot leave a highlight from another page on the
 * overlay, and a zoom re-render maps the stored rectangles with the new
 * viewport.
 */
export function selectPageHighlights(
  selection: HighlightSelection,
  pageNumber: number,
): { hits: AnchorHit[]; activeLocalIndex: number } {
  const hits = selection.hits.filter((hit) => hit.page_number === pageNumber);
  let activeLocalIndex = -1;

  if (selection.activeIndex >= 0) {
    const active = selection.hits[selection.activeIndex];
    if (active && active.page_number === pageNumber) {
      activeLocalIndex = hits.indexOf(active);
    }
  }

  return { hits, activeLocalIndex };
}
