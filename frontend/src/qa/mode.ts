import type { AnswerMode, RetrievalMode } from "../api/types";

export const EVIDENCE_ANSWER_LABEL = "证据展示模式 · 未接入语言模型";
export const GENERATED_ANSWER_LABEL =
  "AI 生成回答 · 引用编号由服务端校验，结论仍需人工核对原文";

export const GROUNDED_CITATION_LIMIT = 12;

export function answerModeLabel(mode: AnswerMode): string {
  return mode === "grounded_generation"
    ? GENERATED_ANSWER_LABEL
    : EVIDENCE_ANSWER_LABEL;
}

/**
 * Grounded generation requests a wider legacy-ranked evidence set so relevant
 * passages ranked below the default Top-5 reach the model. Evidence-only keeps
 * the client's existing default.
 */
export function qaMaxCitations(mode: AnswerMode): number | undefined {
  return mode === "grounded_generation" ? GROUNDED_CITATION_LIMIT : undefined;
}

export function retrievalModeLabel(mode: RetrievalMode): string {
  return mode === "semantic" ? "语义检索" : "关键词检索";
}

/**
 * Map a semantic-retrieval failure to a user-facing message.
 *
 * The semantic path never falls back to literal matching, so these messages
 * explain why no substitute evidence was shown. Returns `null` for literal
 * mode or statuses this helper does not own.
 */
export function semanticFailureMessage(
  status: number,
  mode: RetrievalMode
): string | null {
  if (mode !== "semantic") {
    return null;
  }
  switch (status) {
    case 409:
      return "语义索引缺失或已过期（文档或文本索引已变化）。请先建立或重建语义索引。";
    case 503:
      return "语义检索未配置或未启用。系统未退回关键词检索，也未生成无依据的回答。";
    case 502:
      return "Embedding 服务请求失败或返回无效向量，未展示任何未经校验的回答。";
    case 504:
      return "语义检索请求超时，未展示任何未经校验的回答。";
    default:
      return null;
  }
}

/** Map a semantic-index build failure to a user-facing message. */
export function semanticIndexFailureMessage(status: number): string | null {
  switch (status) {
    case 409:
      return "请先建立文本索引；若文本索引已变化，请重新建立语义索引。";
    case 503:
      return "语义检索未配置或未启用，无法建立语义索引。";
    case 502:
      return "Embedding 服务请求失败或返回无效向量，语义索引未建立。";
    case 504:
      return "建立语义索引超时，原语义索引保持不变。";
    case 500:
      return "语义索引无法保存，原语义索引保持不变。";
    default:
      return null;
  }
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
