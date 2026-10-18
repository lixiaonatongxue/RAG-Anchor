/**
 * Geometry and timing helpers for bringing the active search hit into the
 * center of the PDF scroll container.
 *
 * The helpers intentionally accept structural interfaces rather than concrete
 * DOM types. This keeps the geometry testable in the Node-based Vitest
 * environment while still using the real `getBoundingClientRect()` values in
 * the browser. No PDF user-space conversion or hard-coded pixel offset is
 * performed here.
 */

export interface RectLike {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface ScrollMetrics {
  scrollTop: number;
  scrollLeft: number;
  clientTop: number;
  clientLeft: number;
  clientHeight: number;
  clientWidth: number;
  scrollHeight: number;
  scrollWidth: number;
}

export interface ScrollBoxLike extends ScrollMetrics {
  getBoundingClientRect(): RectLike;
}

export interface ElementLike {
  getBoundingClientRect(): RectLike;
}

export interface ScrollRequestGate {
  /** Start a new scroll request and invalidate all older ones. */
  begin(): number;
  /** Whether the given request is still the newest request. */
  isCurrent(token: number): boolean;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Return the smallest rectangle that contains every input rectangle.
 * Multi-line evidence produces one rectangle per line, so the union is the
 * stable representative area used for centering.
 */
export function unionRects(rects: readonly RectLike[]): RectLike | null {
  if (rects.length === 0) {
    return null;
  }

  let left = Number.POSITIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;

  for (const rect of rects) {
    left = Math.min(left, rect.left);
    top = Math.min(top, rect.top);
    right = Math.max(right, rect.right);
    bottom = Math.max(bottom, rect.bottom);
  }

  return { left, top, right, bottom };
}

/**
 * Compute the scroll offsets that put the target center at the center of the
 * container's client area, clamped to the reachable scroll range.
 *
 * `targetRect` and `containerRect` are `getBoundingClientRect()` values taken
 * before scrolling, so the delta is added to the current scroll offset.
 */
export function computeCenteredScroll(
  containerRect: RectLike,
  metrics: ScrollMetrics,
  targetRect: RectLike
): { scrollLeft: number; scrollTop: number } {
  const containerCenterX =
    containerRect.left + metrics.clientLeft + metrics.clientWidth / 2;
  const containerCenterY =
    containerRect.top + metrics.clientTop + metrics.clientHeight / 2;
  const targetCenterX = (targetRect.left + targetRect.right) / 2;
  const targetCenterY = (targetRect.top + targetRect.bottom) / 2;

  const maxScrollLeft = Math.max(0, metrics.scrollWidth - metrics.clientWidth);
  const maxScrollTop = Math.max(0, metrics.scrollHeight - metrics.clientHeight);

  return {
    scrollLeft: clamp(
      metrics.scrollLeft + targetCenterX - containerCenterX,
      0,
      maxScrollLeft
    ),
    scrollTop: clamp(
      metrics.scrollTop + targetCenterY - containerCenterY,
      0,
      maxScrollTop
    ),
  };
}

/**
 * Scroll the container so the union of the target elements is centered.
 * Returns `false` when there is no target to center.
 */
export function scrollElementsToCenter(
  box: ScrollBoxLike,
  targets: readonly ElementLike[]
): boolean {
  const targetRect = unionRects(
    targets.map((target) => target.getBoundingClientRect())
  );
  if (!targetRect) {
    return false;
  }

  const next = computeCenteredScroll(
    box.getBoundingClientRect(),
    box,
    targetRect
  );
  box.scrollLeft = next.scrollLeft;
  box.scrollTop = next.scrollTop;
  return true;
}

export interface CenterTargetsOptions {
  gate: ScrollRequestGate;
  requestToken: number;
  /** Resolves once the target page finished rendering and painting. */
  wait: () => Promise<void>;
  /** Collect the freshly painted active highlight elements. */
  collectTargets: () => readonly ElementLike[];
  container: ScrollBoxLike;
}

/**
 * Wait for the page render to finish, then center the collected targets.
 *
 * This is the testable core of the cross-page sequence: `wait` resolves only
 * after `viewer.setPage()` rendered the target page and `onRender()` painted
 * the highlight layer, so measurements never come from the previous page.
 * A stale request (a newer click arrived while waiting) or a missing target
 * returns `false` without touching the container.
 */
export async function centerTargetsWhenReady(
  options: CenterTargetsOptions
): Promise<boolean> {
  await options.wait();

  if (!options.gate.isCurrent(options.requestToken)) {
    return false;
  }

  const targets = options.collectTargets();
  if (targets.length === 0) {
    return false;
  }

  return scrollElementsToCenter(options.container, targets);
}

/**
 * Scroll only the given container enough to reveal the target. This is used
 * for the search result list and deliberately avoids `scrollIntoView()`, which
 * can also scroll the window or unrelated ancestors.
 */
export function revealWithin(
  container: ScrollBoxLike,
  target: ElementLike
): void {
  const containerRect = container.getBoundingClientRect();
  const targetRect = target.getBoundingClientRect();

  if (targetRect.top < containerRect.top) {
    container.scrollTop -= containerRect.top - targetRect.top;
  } else if (targetRect.bottom > containerRect.bottom) {
    container.scrollTop += targetRect.bottom - containerRect.bottom;
  }
}

/** Create a monotonic request gate used to discard stale async scrolls. */
export function createScrollRequestGate(): ScrollRequestGate {
  let current = 0;

  return {
    begin(): number {
      current += 1;
      return current;
    },
    isCurrent(token: number): boolean {
      return token === current;
    },
  };
}

/** Resolve on the next animation frame, or immediately without a DOM. */
export function nextFrame(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
      return;
    }
    resolve();
  });
}

/**
 * Wait until the element's running animations have finished and the browser
 * has applied two animation frames of layout. This replaces fixed timeouts
 * when measuring freshly painted highlights.
 */
export async function waitForLayout(element: Element): Promise<void> {
  const animations =
    typeof element.getAnimations === "function"
      ? element
          .getAnimations()
          .filter((animation) => animation.playState === "running")
      : [];

  if (animations.length > 0) {
    await Promise.all(
      animations.map((animation) => animation.finished.catch(() => undefined))
    );
  }

  await nextFrame();
  await nextFrame();
}
