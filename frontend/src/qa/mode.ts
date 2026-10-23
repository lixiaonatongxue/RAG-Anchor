import type { AnswerMode } from "../api/types";

export const EVIDENCE_ANSWER_LABEL = "证据展示模式 · 未接入语言模型";
export const GENERATED_ANSWER_LABEL =
  "AI 生成回答 · 引用编号由服务端校验，结论仍需人工核对原文";

export function answerModeLabel(mode: AnswerMode): string {
  return mode === "grounded_generation"
    ? GENERATED_ANSWER_LABEL
    : EVIDENCE_ANSWER_LABEL;
}

/**
 * Map a grounded-generation failure to a user-facing message.
 *
 * Returns `null` for statuses this helper does not own, so callers can fall
 * back to the API error text. A failed generation never leaves a fake answer
 * or an approximate highlight behind.
 */
export function generationFailureMessage(status: number): string | null {
  switch (status) {
    case 503:
      return "AI 生成模式尚未配置或未启用。可切换到「证据展示」查看原文证据。";
    case 504:
      return "AI 生成请求超时，未收到模型结果，未展示任何生成回答。";
    case 502:
      return "AI 生成失败或返回内容未通过服务端证据校验，未展示任何生成回答。";
    default:
      return null;
  }
}
