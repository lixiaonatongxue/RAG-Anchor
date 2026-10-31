import type {
  AnswerMode,
  QuestionAnswerResponse,
  RetrievalMode,
} from "../api/types";

export const NO_CANDIDATE_REFUSAL = "未找到可用于回答的候选证据。";
export const MODEL_REFUSAL = "已检索到相关候选，但证据不足以回答这个问题。";

/**
 * Explain an `insufficient_evidence` response without assuming that semantic
 * retrieval found nothing. In grounded generation, `total_matches` separates
 * "no candidates were retrieved" from "the model refused despite candidates".
 * Evidence-only wording is unchanged.
 */
export function insufficientEvidenceMessage(
  response: Pick<QuestionAnswerResponse, "total_matches">,
  retrievalMode: RetrievalMode,
  answerMode: AnswerMode
): string {
  if (answerMode === "grounded_generation") {
    return response.total_matches > 0 ? MODEL_REFUSAL : NO_CANDIDATE_REFUSAL;
  }
  return retrievalMode === "semantic"
    ? "语义检索没有找到相似度足够的候选证据，系统已明确拒答。"
    : "没有找到可支持回答的原文证据，已明确拒答。";
}
