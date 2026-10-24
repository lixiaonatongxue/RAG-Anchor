import { describe, expect, it, vi } from "vitest";

import { createRequestGate } from "../src/ui/request";

describe("createRequestGate", () => {
  it("issues strictly increasing tokens", () => {
    const gate = createRequestGate();
    const first = gate.begin();
    const second = gate.begin();
    expect(first).toBeLessThan(second);
    const third = gate.begin();
    expect(second).toBeLessThan(third);
  });

  it("marks every older token as stale once a newer one starts", () => {
    const gate = createRequestGate();
    const first = gate.begin();
    expect(gate.isCurrent(first)).toBe(true);

    const second = gate.begin();
    expect(gate.isCurrent(first)).toBe(false);
    expect(gate.isCurrent(second)).toBe(true);

    const third = gate.begin();
    expect(gate.isCurrent(first)).toBe(false);
    expect(gate.isCurrent(second)).toBe(false);
    expect(gate.isCurrent(third)).toBe(true);
  });

  it("prevents a stale response from overwriting newer request state", async () => {
    const gate = createRequestGate();
    let writeState: string[] = [];
    const apply = (label: string, token: number) => {
      if (gate.isCurrent(token)) {
        writeState.push(label);
      }
    };

    // Simulate two rapid searches: the first is in flight...
    const first = gate.begin();
    const firstDone = Promise.resolve().then(() => apply("first", first));

    // ...the user starts a second search before the first resolves.
    const second = gate.begin();
    const secondDone = Promise.resolve().then(() => apply("second", second));

    // Let the fetch-style microtasks settle in arrival order.
    await firstDone;
    await secondDone;

    // The stale "first" response was ignored by the gate.
    expect(writeState).toEqual(["second"]);
  });

  it("keeps the newest token current across many invocations", () => {
    const gate = createRequestGate();
    let latest = -1;
    for (let index = 0; index < 1000; index += 1) {
      latest = gate.begin();
    }
    expect(gate.isCurrent(latest)).toBe(true);
  });

  it("is usable with a synchronous guard helper via vi.fn caller", () => {
    // Regression guard: main.ts uses `gate.isCurrent(token)` and only
    // commits a response when it matches. Verify a helper built on the
    // gate gives exactly one winner on rapid switching.
    const gate = createRequestGate();
    const callbacks = [vi.fn(), vi.fn(), vi.fn()];
    const tokens = callbacks.map(() => gate.begin());
    tokens.forEach((token, index) => {
      if (gate.isCurrent(token)) {
        callbacks[index]();
      }
    });
    expect(callbacks[0]).not.toHaveBeenCalled();
    expect(callbacks[1]).not.toHaveBeenCalled();
    expect(callbacks[2]).toHaveBeenCalledTimes(1);
  });

  it("invalidates an in-flight answer when a semantic rebuild starts", async () => {
    // Mirrors performSemanticIndex(): starting a rebuild calls
    // qaGate.begin() via resetQaState() before the request is sent.
    const qaGate = createRequestGate();
    const answerToken = qaGate.begin();
    const applied: string[] = [];
    const oldAnswer = Promise.resolve().then(() => {
      if (qaGate.isCurrent(answerToken)) {
        applied.push("old answer");
      }
    });

    const rebuildToken = qaGate.begin();
    await oldAnswer;

    expect(applied).toEqual([]);
    expect(qaGate.isCurrent(answerToken)).toBe(false);
    expect(qaGate.isCurrent(rebuildToken)).toBe(true);
  });

  it("keeps the QA gate usable when a semantic rebuild fails", () => {
    const qaGate = createRequestGate();
    const oldAnswerToken = qaGate.begin();
    const rebuildToken = qaGate.begin();

    // A failed rebuild does not begin a new token; the old answer stays stale
    // and the next question can proceed normally.
    expect(qaGate.isCurrent(oldAnswerToken)).toBe(false);
    expect(qaGate.isCurrent(rebuildToken)).toBe(true);

    const nextAnswerToken = qaGate.begin();
    expect(qaGate.isCurrent(rebuildToken)).toBe(false);
    expect(qaGate.isCurrent(nextAnswerToken)).toBe(true);
  });
});
