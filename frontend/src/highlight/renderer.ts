import type { AnchorHit, RectData } from "../api/types";

/**
 * The subset of PDF.js `PageViewport` used for highlight conversion.
 * Keeping this structural makes the conversion testable without a DOM.
 */
export interface ViewportGeometry {
  readonly width: number;
  readonly height: number;
  convertToViewportRectangle(rect: number[]): number[];
}

export interface OverlayBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export interface HighlightBox extends OverlayBox {
  hitIndex: number;
  rectIndex: number;
  active: boolean;
}

/**
 * Normalize a PDF.js `convertToViewportRectangle` result.
 *
 * PDF user space has its origin at the bottom-left, while CSS positions use
 * the top-left. PDF.js already flips the vertical axis, so the converted
 * values are not necessarily ordered; min/max keeps the box well formed.
 */
export function normalizeConvertedRect(converted: readonly number[]): OverlayBox {
  const [x1, y1, x2, y2] = converted;
  const left = Math.min(x1, x2);
  const right = Math.max(x1, x2);
  const top = Math.min(y1, y2);
  const bottom = Math.max(y1, y2);

  return {
    left,
    top,
    width: right - left,
    height: bottom - top,
  };
}

/**
 * Convert one backend rectangle (absolute PDF user space, points) into an
 * overlay box. No CropBox offset is applied here: the backend already emits
 * absolute coordinates, and PDF.js applies the page rotation and scale.
 */
export function rectToOverlayBox(rect: RectData, viewport: ViewportGeometry): OverlayBox {
  const converted = viewport.convertToViewportRectangle([
    rect.left,
    rect.bottom,
    rect.right,
    rect.top,
  ]);
  return normalizeConvertedRect(converted);
}

/**
 * Build one box per backend rectangle. Multi-line evidence stays as several
 * separate row boxes instead of one merged bounding box.
 */
export function buildHighlightBoxes(
  hits: readonly AnchorHit[],
  viewport: ViewportGeometry,
  activeHitIndex = -1,
): HighlightBox[] {
  const boxes: HighlightBox[] = [];

  hits.forEach((hit, hitIndex) => {
    hit.rects.forEach((rect, rectIndex) => {
      const box = rectToOverlayBox(rect, viewport);
      if (box.width <= 0 || box.height <= 0) {
        return;
      }
      boxes.push({
        ...box,
        hitIndex,
        rectIndex,
        active: hitIndex === activeHitIndex,
      });
    });
  });

  return boxes;
}

/** Draw all highlights for the visible page into the overlay layer. */
export function paintHighlights(
  layer: HTMLElement,
  hits: readonly AnchorHit[],
  viewport: ViewportGeometry,
  activeHitIndex = -1,
): void {
  layer.replaceChildren();
  layer.style.width = `${viewport.width}px`;
  layer.style.height = `${viewport.height}px`;

  const boxes = buildHighlightBoxes(hits, viewport, activeHitIndex);
  const fragment = document.createDocumentFragment();

  for (const box of boxes) {
    const element = document.createElement("div");
    element.className = box.active
      ? "pdf-highlight pdf-highlight--active"
      : "pdf-highlight";
    element.style.left = `${box.left}px`;
    element.style.top = `${box.top}px`;
    element.style.width = `${box.width}px`;
    element.style.height = `${box.height}px`;
    element.dataset.hitIndex = String(box.hitIndex);
    element.dataset.rectIndex = String(box.rectIndex);
    fragment.append(element);
  }

  layer.append(fragment);
}
