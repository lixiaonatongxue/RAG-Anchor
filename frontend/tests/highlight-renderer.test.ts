import { describe, expect, it } from "vitest";

import type { AnchorHit, RectData } from "../src/api/types";
import {
  buildHighlightBoxes,
  normalizeConvertedRect,
  rectToOverlayBox,
} from "../src/highlight/renderer";
import type { ViewportGeometry } from "../src/highlight/renderer";

/**
 * A structural stand-in for a PDF.js PageViewport.
 *
 * PDF.js maps PDF user space (origin bottom-left) into viewport space
 * (origin top-left) using `height - top` for the vertical axis. This mock
 * reproduces that behaviour so the tests exercise real conversion math
 * without needing a DOM or a real PDF.
 */
function makeViewport(scale: number, pageHeightPt = 400): ViewportGeometry {
  return {
    width: 300 * scale,
    height: pageHeightPt * scale,
    convertToViewportRectangle([left, bottom, right, top]: number[]): number[] {
      return [
        left * scale,
        (pageHeightPt - top) * scale,
        right * scale,
        (pageHeightPt - bottom) * scale,
      ];
    },
  };
}

function makeRect(overrides: Partial<RectData> = {}): RectData {
  return { left: 100, bottom: 200, right: 120, top: 220, ...overrides };
}

function makeHit(rects: RectData[], pageNumber = 1): AnchorHit {
  return {
    page_index: pageNumber - 1,
    page_number: pageNumber,
    matched_text: "evidence",
    char_start: 0,
    char_count: 8,
    rects,
    bbox: null,
    page: {
      page_index: pageNumber - 1,
      width_pt: 300,
      height_pt: 400,
      rotation: 0,
      cropbox: [0, 0, 300, 400],
    },
  };
}

describe("normalizeConvertedRect", () => {
  it("computes left/top/width/height from an ordered rectangle", () => {
    expect(normalizeConvertedRect([100, 200, 120, 220])).toEqual({
      left: 100,
      top: 200,
      width: 20,
      height: 20,
    });
  });

  it("normalizes a rectangle whose x/y axes are reversed", () => {
    expect(normalizeConvertedRect([120, 220, 100, 200])).toEqual({
      left: 100,
      top: 200,
      width: 20,
      height: 20,
    });
  });
});

describe("rectToOverlayBox", () => {
  it("converts PDF user-space points through the viewport at scale 1", () => {
    const box = rectToOverlayBox(
      makeRect({ left: 100, bottom: 200, right: 107.164, top: 208.616 }),
      makeViewport(1),
    );

    expect(box.left).toBeCloseTo(100, 6);
    expect(box.top).toBeCloseTo(400 - 208.616, 6);
    expect(box.width).toBeCloseTo(7.164, 6);
    expect(box.height).toBeCloseTo(8.616, 6);
  });

  it("scales width and height proportionally with the viewport", () => {
    const rect = makeRect();
    const at1 = rectToOverlayBox(rect, makeViewport(1));
    const at2 = rectToOverlayBox(rect, makeViewport(2));

    expect(at2.width).toBeCloseTo(at1.width * 2, 6);
    expect(at2.height).toBeCloseTo(at1.height * 2, 6);
    expect(at2.left).toBeCloseTo(at1.left * 2, 6);
    expect(at2.top).toBeCloseTo(at1.top * 2, 6);
  });

  it("does not subtract a CropBox origin", () => {
    // ADR-0001: the backend already emits absolute PDF user-space values.
    const box = rectToOverlayBox(
      makeRect({ left: 124.432, bottom: 197.528, right: 152.032, top: 208.616 }),
      makeViewport(1),
    );

    expect(box.left).toBeCloseTo(124.432, 6);
  });
});

describe("buildHighlightBoxes", () => {
  it("keeps multi-line evidence as separate row boxes", () => {
    const hit = makeHit([
      { left: 201.22, bottom: 197.372, right: 240.436, top: 206.456 },
      { left: 100.384, bottom: 149.832, right: 138.004, top: 158.616 },
    ]);

    const boxes = buildHighlightBoxes([hit], makeViewport(1));

    expect(boxes).toHaveLength(2);
    expect(boxes[0].rectIndex).toBe(0);
    expect(boxes[1].rectIndex).toBe(1);
    expect(boxes[0].top).not.toBeCloseTo(boxes[1].top, 6);
    expect(boxes[0].width).toBeCloseTo(39.216, 6);
    expect(boxes[1].width).toBeCloseTo(37.62, 6);
  });

  it("filters out zero-area rectangles", () => {
    const hit = makeHit([
      makeRect({ left: 100, right: 100 }),
      makeRect({ left: 100, right: 120 }),
    ]);

    const boxes = buildHighlightBoxes([hit], makeViewport(1));

    expect(boxes).toHaveLength(1);
  });

  it("marks only the active hit's boxes as active", () => {
    const hits = [
      makeHit([makeRect()], 1),
      makeHit([makeRect({ left: 140, right: 160 })], 1),
    ];

    const boxes = buildHighlightBoxes(hits, makeViewport(1), 1);

    expect(boxes).toHaveLength(2);
    expect(boxes[0].active).toBe(false);
    expect(boxes[0].hitIndex).toBe(0);
    expect(boxes[1].active).toBe(true);
    expect(boxes[1].hitIndex).toBe(1);
  });

  it("returns an empty list when there are no hits", () => {
    expect(buildHighlightBoxes([], makeViewport(1))).toEqual([]);
  });
});
