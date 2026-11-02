import { describe, expect, it } from "vitest";

import {
  EVIDENCE_ANSWER_LABEL,
  GENERATED_ANSWER_LABEL,
  GROUNDED_CITATION_LIMIT,
  answerModeLabel,
  generationFailureMessage,
  qaMaxCitations,
  retrievalModeLabel,
  semanticFailureMessage,
  semanticIndexFailureMessage,
} from "../src/qa/mode";

describe("answer mode labels", () => {
  it("labels evidence mode as not using a language model", () => {
    expect(answerModeLabel("evidence_only")).toBe(EVIDENCE_ANSWER_LABEL);
    expect(EVIDENCE_ANSWER_LABEL).toContain("未接入语言模型");
  });

  it("labels grounded mode without claiming the answer itself is verified", () => {
    expect(answerModeLabel("grounded_generation")).toBe(
      GENERATED_ANSWER_LABEL
    );
    expect(GENERATED_ANSWER_LABEL).toContain("服务端校验");
    expect(GENERATED_ANSWER_LABEL).toContain("人工核对");
    expect(GENERATED_ANSWER_LABEL).not.toContain("已验证结论");
  });
});

describe("grounded evidence limit", () => {
  it("requests twelve citations for grounded generation", () => {
    expect(GROUNDED_CITATION_LIMIT).toBe(12);
    expect(qaMaxCitations("grounded_generation")).toBe(12);
  });

  it("keeps evidence-only on the existing client default", () => {
    expect(qaMaxCitations("evidence_only")).toBeUndefined();
  });
});

describe("generationFailureMessage", () => {
  it("explains that grounded generation is unavailable without faking success", () => {
    const message = generationFailureMessage(503);
    expect(message).toContain("尚未配置或未启用");
    expect(message).toContain("证据展示");
  });

  it("explains timeouts without showing model output", () => {
    const message = generationFailureMessage(504);
    expect(message).toContain("超时");
    expect(message).toContain("未展示任何生成回答");
  });

  it("explains verification failures without showing model output", () => {
    const message = generationFailureMessage(502);
    expect(message).toContain("证据校验");
    expect(message).toContain("未展示任何生成回答");
  });

  it("leaves unrelated statuses to the generic error path", () => {
    expect(generationFailureMessage(500)).toBeNull();
    expect(generationFailureMessage(0)).toBeNull();
    expect(generationFailureMessage(404)).toBeNull();
  });
});

describe("retrieval mode labels and semantic failures", () => {
  it("labels both retrieval layers distinctly", () => {
    expect(retrievalModeLabel("literal")).toBe("关键词检索");
    expect(retrievalModeLabel("semantic")).toBe("语义检索");
  });

  it("does not replace a semantic failure with literal retrieval", () => {
    const missingIndex = semanticFailureMessage(409, "semantic");
    expect(missingIndex).toContain("语义索引");
    expect(missingIndex).toContain("重建");
    expect(missingIndex).not.toContain("字面匹配到");

    const disabled = semanticFailureMessage(503, "semantic");
    expect(disabled).toContain("未启用");
    expect(disabled).toContain("未退回关键词检索");

    const timeout = semanticFailureMessage(504, "semantic");
    expect(timeout).toContain("超时");
    expect(timeout).toContain("未展示");
  });

  it("never maps a literal failure to a semantic message", () => {
    expect(semanticFailureMessage(409, "literal")).toBeNull();
    expect(semanticFailureMessage(503, "literal")).toBeNull();
  });

  it("explains semantic index build failures without claiming success", () => {
    expect(semanticIndexFailureMessage(409)).toContain("文本索引");
    expect(semanticIndexFailureMessage(503)).toContain("无法建立");
    expect(semanticIndexFailureMessage(502)).toContain("无效向量");
    expect(semanticIndexFailureMessage(504)).toContain("超时");
    expect(semanticIndexFailureMessage(200)).toBeNull();
  });
});
