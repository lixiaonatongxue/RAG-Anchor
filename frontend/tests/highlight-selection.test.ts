import { describe, expect, it } from "vitest";

import {
  anchorSelection,
  emptySelection,
  evidenceSelection,
  locatedEvidenceToHit,
  selectPageHighlights,
} from "../src/highlight/selection";
import type { AnchorHit, LocatedEvidence } from "../src/api/types";

const pageA = {
  page_index: 0,
  width_pt: 612,
  height_pt: 792,
  rotation: 0,
  cropbox: [0, 0, 612, 792] as [number, number, number, number],
};

function anchorHit(overrides: Partial<AnchorHit> = {}): AnchorHit {
  return {
    page_index: 0,
    page_number: 1,
    matched_text: "quick",
    char_start: 4,
    char_count: 5,
    rects: [{ left: 100, bottom: 100, right: 110, top: 110 }],
    bbox: { left: 100, bottom: 100, right: 110, top: 110 },
    page: pageA,
    ...overrides,
  };
}

const located: LocatedEvidence = {
  page_index: 3,
  page_number: 4,
  match_start: 12,
  match_end: 17,
  hit_text: "quick",
  chunk_id: "chunk-0001",
  char_start: 40,
  char_count: 5,
  matched_text: "quick",
  rects: [
    { left: 124.432, bottom: 197.528, right: 152.032, top: 208.616 },
    { left: 124.432, bottom: 211.416, right: 156.202, top: 220.802 },
  ],
  bbox: { left: 124.432, bottom: 197.528, right: 156.202, top: 220.802 },
  page: {
    page_index: 3,
    width_pt: 612,
    height_pt: 792,
    rotation: 0,
    cropbox: [0, 0, 612, 792],
  },
};

describe("emptySelection / anchorSelection", () => {
  it("starts with no hits and no active index", () => {
    expect(emptySelection()).toEqual({ hits: [], activeIndex: -1 });
  });

  it("normalizes a valid active index into a copied hit list", () => {
    const hits = [anchorHit(), anchorHit({ matched_text: "brown" })];
    const selection = anchorSelection(hits, 1);

    expect(selection).toEqual({ hits, activeIndex: 1 });
    expect(selection.hits).not.toBe(hits);
  });

  it("normalizes an out-of-range active index to -1", () => {
    const hits = [anchorHit()];
    expect(anchorSelection(hits, 5).activeIndex).toBe(-1);
    expect(anchorSelection([], 0).activeIndex).toBe(-1);
    expect(anchorSelection(hits, -1).activeIndex).toBe(-1);
  });

  it("keeps the active index when the hit list is long enough", () => {
    const hits = [anchorHit(), anchorHit(), anchorHit({ matched_text: "fox" })];
    expect(anchorSelection(hits, 2).activeIndex).toBe(2);
  });
});

describe("locatedEvidenceToHit", () => {
  it("keeps the server rect objects and page descriptor by reference", () => {
    const converted = locatedEvidenceToHit(located);
    expect(converted.rects).toBe(located.rects);
    expect(converted.bbox).toBe(located.bbox);
    expect(converted.page).toBe(located.page);
  });

  it("copies only the fields the renderer needs", () => {
    const converted = locatedEvidenceToHit(located);
    expect(converted).toEqual({
      page_index: 3,
      page_number: 4,
      matched_text: "quick",
      char_start: 40,
      char_count: 5,
      rects: located.rects,
      bbox: located.bbox,
      page: located.page,
    });
  });

  it("drops the request-side span fields", () => {
    const converted = locatedEvidenceToHit(located);
    expect(converted).not.toHaveProperty("match_start");
    expect(converted).not.toHaveProperty("match_end");
    expect(converted).not.toHaveProperty("hit_text");
    expect(converted).not.toHaveProperty("chunk_id");
  });
});

describe("evidenceSelection", () => {
  it("wraps a located evidence as the only active hit", () => {
    const selection = evidenceSelection(located);
    expect(selection.activeIndex).toBe(0);
    expect(selection.hits).toHaveLength(1);
    expect(selection.hits[0].rects).toBe(located.rects);
  });

  it("returns an empty selection for a missing result", () => {
    expect(evidenceSelection(null)).toEqual({ hits: [], activeIndex: -1 });
  });
});

describe("selectPageHighlights", () => {
  it("returns nothing for an empty selection", () => {
    expect(selectPageHighlights(emptySelection(), 1)).toEqual({
      hits: [],
      activeLocalIndex: -1,
    });
  });

  it("filters hits to the rendered page and drops the rest", () => {
    const selection = anchorSelection(
      [
        anchorHit({ page_number: 1, matched_text: "quick" }),
        anchorHit({ page_number: 2, page_index: 1, matched_text: "brown" }),
        anchorHit({ page_number: 1, matched_text: "lazy" }),
      ],
      2
    );

    const pageOne = selectPageHighlights(selection, 1);
    expect(pageOne.hits.map((item) => item.matched_text)).toEqual([
      "quick",
      "lazy",
    ]);
    // Active (index 2 = "lazy") is on page 1, so it is found locally.
    expect(pageOne.activeLocalIndex).toBe(1);

    const pageTwo = selectPageHighlights(selection, 2);
    expect(pageTwo.hits.map((item) => item.matched_text)).toEqual(["brown"]);
    expect(pageTwo.activeLocalIndex).toBe(-1);
  });

  it("keeps a single multi-rect hit on its page with local index 0", () => {
    const selection = evidenceSelection(located);
    const onPageFour = selectPageHighlights(selection, 4);
    expect(onPageFour.hits).toHaveLength(1);
    expect(onPageFour.hits[0].rects).toHaveLength(2);
    expect(onPageFour.activeLocalIndex).toBe(0);
  });

  it("switching pages cannot leave a highlight from another page", () => {
    const selection = evidenceSelection(located);
    const elsewhere = selectPageHighlights(selection, 1);
    expect(elsewhere.hits).toEqual([]);
    expect(elsewhere.activeLocalIndex).toBe(-1);
  });
});
