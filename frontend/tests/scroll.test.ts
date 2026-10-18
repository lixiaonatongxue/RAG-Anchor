import { describe, expect, it, vi } from "vitest";

import {
  centerTargetsWhenReady,
  computeCenteredScroll,
  createScrollRequestGate,
  nextFrame,
  revealWithin,
  scrollElementsToCenter,
  unionRects,
  waitForLayout,
} from "../src/ui/scroll";
import type { ElementLike, RectLike, ScrollBoxLike } from "../src/ui/scroll";

function rect(
  left: number,
  top: number,
  right: number,
  bottom: number,
): RectLike {
  return { left, top, right, bottom };
}

function makeBox(overrides: Partial<ScrollBoxLike> = {}): ScrollBoxLike {
  return {
    scrollTop: 0,
    scrollLeft: 0,
    clientTop: 0,
    clientLeft: 0,
    clientHeight: 400,
    clientWidth: 300,
    scrollHeight: 1200,
    scrollWidth: 600,
    getBoundingClientRect: () => rect(0, 0, 300, 400),
    ...overrides,
  };
}

function target(area: RectLike): ElementLike {
  return { getBoundingClientRect: () => area };
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

describe("unionRects", () => {
  it("returns null when there is nothing to measure", () => {
    expect(unionRects([])).toBeNull();
  });

  it("returns the single rectangle unchanged", () => {
    expect(unionRects([rect(10, 20, 30, 40)])).toEqual({
      left: 10,
      top: 20,
      right: 30,
      bottom: 40,
    });
  });

  it("covers every line of a multi-line hit", () => {
    expect(
      unionRects([rect(0, 100, 10, 110), rect(50, 300, 60, 320)]),
    ).toEqual({
      left: 0,
      top: 100,
      right: 60,
      bottom: 320,
    });
  });
});

describe("computeCenteredScroll", () => {
  it("scrolls down when the same-page hit is below the viewport", () => {
    const box = makeBox();
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(100, 800, 140, 820),
    );

    // Target center Y = 810, container center Y = 200, so scrollTop = 610.
    expect(result.scrollTop).toBe(610);
    expect(result.scrollLeft).toBe(0);
  });

  it("scrolls up when the same-page hit is above the viewport", () => {
    const box = makeBox({ scrollTop: 300 });
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(100, -100, 140, -80),
    );

    // Target center Y = -90; 300 + (-90 - 200) = 10.
    expect(result.scrollTop).toBe(10);
  });

  it("recenters an already visible hit without drifting", () => {
    const box = makeBox({ scrollTop: 300, scrollLeft: 100 });
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(130, 180, 170, 220),
    );

    // Target center equals the container client center, so the offsets stay.
    expect(result).toEqual({ scrollLeft: 100, scrollTop: 300 });
  });

  it("clamps at the top edge", () => {
    const box = makeBox();
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(0, -600, 10, -580),
    );

    expect(result.scrollTop).toBe(0);
  });

  it("clamps at the bottom edge", () => {
    const box = makeBox({ scrollTop: 800 });
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(0, 3800, 10, 3820),
    );

    // maxScrollTop = 1200 - 400 = 800.
    expect(result.scrollTop).toBe(800);
  });

  it("also centers horizontally when the page overflows", () => {
    const box = makeBox();
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(300, 100, 340, 140),
    );

    expect(result.scrollLeft).toBe(170);
    expect(result.scrollTop).toBe(0);
  });

  it("accounts for container border offsets", () => {
    const box = makeBox({
      clientLeft: 5,
      clientTop: 5,
      getBoundingClientRect: () => rect(10, 20, 310, 420),
    });
    const result = computeCenteredScroll(
      box.getBoundingClientRect(),
      box,
      rect(155, 215, 175, 235),
    );

    expect(result).toEqual({ scrollLeft: 0, scrollTop: 0 });
  });

  it("scales with zoom instead of using fixed pixel offsets", () => {
    const small = makeBox();
    const large = makeBox({
      clientWidth: 600,
      clientHeight: 800,
      scrollWidth: 1200,
      scrollHeight: 2400,
      getBoundingClientRect: () => rect(0, 0, 600, 800),
    });

    const smallResult = computeCenteredScroll(
      small.getBoundingClientRect(),
      small,
      rect(180, 380, 220, 420),
    );
    const largeResult = computeCenteredScroll(
      large.getBoundingClientRect(),
      large,
      rect(360, 760, 440, 840),
    );

    expect(smallResult).toEqual({ scrollLeft: 50, scrollTop: 200 });
    expect(largeResult).toEqual({ scrollLeft: 100, scrollTop: 400 });
  });
});

describe("scrollElementsToCenter", () => {
  it("returns false and leaves the container untouched without targets", () => {
    const box = makeBox({ scrollTop: 120 });
    expect(scrollElementsToCenter(box, [])).toBe(false);
    expect(box.scrollTop).toBe(120);
  });

  it("centers the union of several target boxes", () => {
    const box = makeBox();
    const result = scrollElementsToCenter(box, [
      target(rect(0, 100, 10, 110)),
      target(rect(50, 300, 60, 320)),
    ]);

    expect(result).toBe(true);
    // Union center = (30, 210); container center = (150, 200).
    expect(box.scrollTop).toBe(10);
    expect(box.scrollLeft).toBe(0);
  });

  it("centers a single multi-line target area", () => {
    const box = makeBox();
    const result = scrollElementsToCenter(box, [
      target(rect(100, 780, 240, 800)),
      target(rect(100, 820, 180, 840)),
    ]);

    expect(result).toBe(true);
    // Union center Y = 810, so the same offset as one 800..820 box.
    expect(box.scrollTop).toBe(610);
  });
});

describe("revealWithin", () => {
  it("does not scroll when the target is already visible", () => {
    const box = makeBox({ scrollTop: 100 });
    revealWithin(box, target(rect(100, 100, 200, 300)));
    expect(box.scrollTop).toBe(100);
  });

  it("scrolls up just enough for a target above the viewport", () => {
    const box = makeBox({ scrollTop: 100 });
    revealWithin(box, target(rect(10, -50, 50, -30)));
    expect(box.scrollTop).toBe(50);
  });

  it("scrolls down just enough for a target below the viewport", () => {
    const box = makeBox({ scrollTop: 100 });
    revealWithin(box, target(rect(10, 380, 50, 430)));
    expect(box.scrollTop).toBe(130);
  });
});

describe("createScrollRequestGate", () => {
  it("invalidates older requests as soon as a new one starts", () => {
    const gate = createScrollRequestGate();

    const first = gate.begin();
    expect(gate.isCurrent(first)).toBe(true);

    const second = gate.begin();
    expect(first).toBeLessThan(second);
    expect(gate.isCurrent(first)).toBe(false);
    expect(gate.isCurrent(second)).toBe(true);
  });
});

describe("nextFrame / waitForLayout", () => {
  it("resolves nextFrame without a DOM", async () => {
    await expect(nextFrame()).resolves.toBeUndefined();
  });

  it("resolves immediately when the element has no animations", async () => {
    const element = {
      getAnimations: () => [],
    } as unknown as Element;

    await expect(waitForLayout(element)).resolves.toBeUndefined();
  });

  it("waits for running animations such as the page-in transform", async () => {
    const finished = deferred();
    const element = {
      getAnimations: () => [{ playState: "running", finished: finished.promise }],
    } as unknown as Element;

    let settled = false;
    const pending = waitForLayout(element).then(() => {
      settled = true;
    });

    await Promise.resolve();
    expect(settled).toBe(false);

    finished.resolve();
    await pending;
    expect(settled).toBe(true);
  });

  it("ignores animations that already finished", async () => {
    const element = {
      getAnimations: () => [
        { playState: "finished", finished: new Promise<void>(() => undefined) },
      ],
    } as unknown as Element;

    await expect(waitForLayout(element)).resolves.toBeUndefined();
  });
});

describe("centerTargetsWhenReady", () => {
  it("waits for the cross-page render before measuring and centering", async () => {
    const gate = createScrollRequestGate();
    const requestToken = gate.begin();
    const renderDone = deferred();
    const box = makeBox();
    const collectTargets = vi.fn(() => [target(rect(100, 800, 140, 820))]);

    const pending = centerTargetsWhenReady({
      gate,
      requestToken,
      wait: () => renderDone.promise,
      collectTargets,
      container: box,
    });

    await Promise.resolve();
    expect(collectTargets).not.toHaveBeenCalled();
    expect(box.scrollTop).toBe(0);

    renderDone.resolve();
    await expect(pending).resolves.toBe(true);
    expect(collectTargets).toHaveBeenCalledTimes(1);
    expect(box.scrollTop).toBe(610);
  });

  it("abandons a request superseded while the page was rendering", async () => {
    const gate = createScrollRequestGate();
    const requestToken = gate.begin();
    const renderDone = deferred();
    const box = makeBox();
    const collectTargets = vi.fn(() => [target(rect(100, 800, 140, 820))]);

    const pending = centerTargetsWhenReady({
      gate,
      requestToken,
      wait: () => renderDone.promise,
      collectTargets,
      container: box,
    });

    gate.begin();
    renderDone.resolve();

    await expect(pending).resolves.toBe(false);
    expect(collectTargets).not.toHaveBeenCalled();
    expect(box.scrollTop).toBe(0);
  });

  it("only the last of several rapid clicks scrolls", async () => {
    const gate = createScrollRequestGate();
    const box = makeBox();

    const firstToken = gate.begin();
    const firstRender = deferred();
    const firstCollect = vi.fn(() => [target(rect(0, 3800, 10, 3820))]);
    const first = centerTargetsWhenReady({
      gate,
      requestToken: firstToken,
      wait: () => firstRender.promise,
      collectTargets: firstCollect,
      container: box,
    });

    const secondToken = gate.begin();
    const secondRender = deferred();
    const secondCollect = vi.fn(() => [target(rect(100, 800, 140, 820))]);
    const second = centerTargetsWhenReady({
      gate,
      requestToken: secondToken,
      wait: () => secondRender.promise,
      collectTargets: secondCollect,
      container: box,
    });

    // Resolve the newest request first, then the stale one. If the stale
    // request were not gated it would overwrite the newer scroll position.
    secondRender.resolve();
    await expect(second).resolves.toBe(true);
    expect(box.scrollTop).toBe(610);

    firstRender.resolve();
    await expect(first).resolves.toBe(false);
    expect(firstCollect).not.toHaveBeenCalled();
    expect(box.scrollTop).toBe(610);
  });

  it("returns false when the active highlight was not painted", async () => {
    const gate = createScrollRequestGate();
    const requestToken = gate.begin();
    const box = makeBox();

    await expect(
      centerTargetsWhenReady({
        gate,
        requestToken,
        wait: () => Promise.resolve(),
        collectTargets: () => [],
        container: box,
      }),
    ).resolves.toBe(false);

    expect(box.scrollTop).toBe(0);
    expect(box.scrollLeft).toBe(0);
  });
});
