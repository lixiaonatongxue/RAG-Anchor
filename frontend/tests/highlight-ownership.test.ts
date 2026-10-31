import { describe, expect, it } from "vitest";

import type { AnchorHit } from "../src/api/types";
import {
  clearHighlightFor,
  emptyHighlightState,
  highlightFor,
} from "../src/highlight/ownership";
import { anchorSelection } from "../src/highlight/selection";

const hit: AnchorHit = {
  page_index: 0,
  page_number: 1,
  matched_text: "narwhal",
  char_start: 0,
  char_count: 7,
  rects: [{ left: 100, bottom: 100, right: 140, top: 110 }],
  bbox: { left: 100, bottom: 100, right: 140, top: 110 },
  page: {
    page_index: 0,
    width_pt: 612,
    height_pt: 792,
    rotation: 0,
    cropbox: [0, 0, 612, 792],
  },
};

const selected = anchorSelection([hit], 0);

describe("highlight ownership", () => {
  it("starts with no source and no hits", () => {
    expect(emptyHighlightState()).toEqual({
      selection: { hits: [], activeIndex: -1 },
      source: null,
    });
  });

  it("clears a selected QA citation when a new QA request starts", () => {
    const e1 = highlightFor("qa", selected);

    const afterNewRequest = clearHighlightFor(e1, "qa");

    expect(afterNewRequest.source).toBeNull();
    expect(afterNewRequest.selection.hits).toEqual([]);
    expect(afterNewRequest.selection.activeIndex).toBe(-1);
  });

  it("keeps a literal-search highlight when QA cleanup runs", () => {
    const anchor = highlightFor("anchor", selected);

    expect(clearHighlightFor(anchor, "qa")).toBe(anchor);
  });

  it("keeps a chunk-evidence highlight when QA cleanup runs", () => {
    const chunk = highlightFor("chunk", selected);

    expect(clearHighlightFor(chunk, "qa")).toBe(chunk);
  });

  it("clears any highlight source on a document switch", () => {
    const qa = highlightFor("qa", selected);
    const anchor = highlightFor("anchor", selected);

    expect(clearHighlightFor(qa).source).toBeNull();
    expect(clearHighlightFor(anchor).source).toBeNull();
  });

  it("replaces the previous source when a new highlight is selected", () => {
    const qa = highlightFor("qa", selected);
    const anchor = highlightFor("anchor", anchorSelection([hit], 0));

    expect(qa.source).toBe("qa");
    expect(anchor.source).toBe("anchor");
    expect(anchor.selection.hits).toHaveLength(1);
  });
});
