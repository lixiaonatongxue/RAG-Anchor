import type {
  EvidenceCitation,
  EvidenceSpanRequest,
} from "../api/types";

/**
 * Build one evidence/locate item from a question citation.
 *
 * The span comes straight from the server response. Clicking a citation must
 * never search `hit_text` again, and no PDF coordinates are derived here.
 */
export function toCitationEvidenceSpan(
  citation: EvidenceCitation
): EvidenceSpanRequest {
  return {
    page_index: citation.page_index,
    match_start: citation.match_start,
    match_end: citation.match_end,
    hit_text: citation.hit_text,
    chunk_id: citation.chunk_id,
  };
}

/** Citations are only actionable inside the document they were retrieved from. */
export function citationBelongsToDocument(
  citation: EvidenceCitation,
  documentId: string
): boolean {
  return citation.document_id === documentId;
}

/** Compact label for a citation button. */
export function citationLabel(citation: EvidenceCitation): string {
  return `${citation.citation_id} · 第 ${citation.page_number} 页 · 字符 ${citation.match_start}–${citation.match_end}`;
}
