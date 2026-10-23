import { describe, expect, it } from "vitest";

import {
  EVIDENCE_ANSWER_LABEL,
  GENERATED_ANSWER_LABEL,
  answerModeLabel,
  generationFailureMessage,
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
