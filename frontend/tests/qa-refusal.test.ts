import { describe, expect, it } from "vitest";

import {
  MODEL_REFUSAL,
  NO_CANDIDATE_REFUSAL,
  insufficientEvidenceMessage,
} from "../src/qa/refusal";

function response(total_matches: number) {
  return { total_matches };
}

describe("insufficientEvidenceMessage", () => {
  it("reports no candidates when grounded generation retrieved none", () => {
    const message = insufficientEvidenceMessage(
      response(0),
      "semantic",
      "grounded_generation"
    );

    expect(message).toBe(NO_CANDIDATE_REFUSAL);
    expect(message).toBe("未找到可用于回答的候选证据。");
    expect(message).not.toContain("相似度足够");
  });

  it("reports a model refusal when candidates were retrieved", () => {
    const message = insufficientEvidenceMessage(
      response(3),
      "semantic",
      "grounded_generation"
    );

    expect(message).toBe(MODEL_REFUSAL);
    expect(message).toBe("已检索到相关候选，但证据不足以回答这个问题。");
    expect(message).not.toContain("没有找到");
  });

  it("does not let semantic-retrieval wording override a model refusal", () => {
    const semantic = insufficientEvidenceMessage(
      response(5),
      "semantic",
      "grounded_generation"
    );
    const literal = insufficientEvidenceMessage(
      response(5),
      "literal",
      "grounded_generation"
    );

    expect(semantic).toBe(literal);
    expect(semantic).toBe(MODEL_REFUSAL);
  });

  it("preserves the evidence-only semantic refusal wording", () => {
    expect(
      insufficientEvidenceMessage(response(0), "semantic", "evidence_only")
    ).toBe("语义检索没有找到相似度足够的候选证据，系统已明确拒答。");
  });

  it("preserves the evidence-only literal refusal wording", () => {
    expect(
      insufficientEvidenceMessage(response(0), "literal", "evidence_only")
    ).toBe("没有找到可支持回答的原文证据，已明确拒答。");
  });

  it("keeps evidence-only wording even if a candidate count is present", () => {
    expect(
      insufficientEvidenceMessage(response(2), "semantic", "evidence_only")
    ).toBe("语义检索没有找到相似度足够的候选证据，系统已明确拒答。");
  });
});
