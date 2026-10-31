import { describe, expect, it } from "vitest";

import { isCurrentQaResponse } from "../src/qa/request";
import type { QaRequestContext, QaUiContext } from "../src/qa/request";
import { createRequestGate } from "../src/ui/request";

const request: QaRequestContext = {
  documentId: "doc-a",
  answerMode: "grounded_generation",
  retrievalMode: "semantic",
};

function ui(overrides: Partial<QaUiContext> = {}): QaUiContext {
  return {
    documentId: "doc-a",
    answerMode: "grounded_generation",
    retrievalMode: "semantic",
    ...overrides,
  };
}

describe("isCurrentQaResponse", () => {
  it("accepts the newest response for the same document and modes", () => {
    const gate = createRequestGate();
    const token = gate.begin();

    expect(isCurrentQaResponse(gate, token, request, ui())).toBe(true);
  });

  it("rejects an older response after a rapid second request", () => {
    const gate = createRequestGate();
    const first = gate.begin();
    gate.begin();

    expect(isCurrentQaResponse(gate, first, request, ui())).toBe(false);
  });

  it("rejects a response after switching documents", () => {
    const gate = createRequestGate();
    const token = gate.begin();

    expect(
      isCurrentQaResponse(gate, token, request, ui({ documentId: "doc-b" }))
    ).toBe(false);
    expect(
      isCurrentQaResponse(gate, token, request, ui({ documentId: null }))
    ).toBe(false);
  });

  it("rejects a response after changing answer mode", () => {
    const gate = createRequestGate();
    const token = gate.begin();

    expect(
      isCurrentQaResponse(
        gate,
        token,
        request,
        ui({ answerMode: "evidence_only" })
      )
    ).toBe(false);
  });

  it("rejects a response after changing retrieval mode", () => {
    const gate = createRequestGate();
    const token = gate.begin();

    expect(
      isCurrentQaResponse(gate, token, request, ui({ retrievalMode: "literal" }))
    ).toBe(false);
  });

  it("accepts a newer response after the earlier one was superseded", () => {
    const gate = createRequestGate();
    gate.begin();
    const newest = gate.begin();

    expect(isCurrentQaResponse(gate, newest, request, ui())).toBe(true);
  });
});
