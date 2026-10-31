import type { AnswerMode, RetrievalMode } from "../api/types";
import type { RequestGate } from "../ui/request";

export interface QaRequestContext {
  documentId: string;
  answerMode: AnswerMode;
  retrievalMode: RetrievalMode;
}

export interface QaUiContext {
  documentId: string | null;
  answerMode: AnswerMode;
  retrievalMode: RetrievalMode;
}

/**
 * A response may only write QA UI state when its request is still the newest
 * one and the document or mode selections have not changed.
 */
export function isCurrentQaResponse(
  gate: RequestGate,
  token: number,
  request: QaRequestContext,
  ui: QaUiContext
): boolean {
  return (
    gate.isCurrent(token) &&
    ui.documentId === request.documentId &&
    ui.answerMode === request.answerMode &&
    ui.retrievalMode === request.retrievalMode
  );
}
