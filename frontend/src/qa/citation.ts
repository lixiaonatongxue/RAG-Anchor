import type {
  CitationSupportKind,
  EvidenceCitation,
  EvidenceSpanRequest,
} from "../api/types";

/** The exact page-text span a citation click should locate. */
export interface CitationSupportSpan {
  text: string;
  start: number;
  end: number;
  kind: CitationSupportKind;
}

function isInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value);
}

function supportKind(citation: EvidenceCitation): CitationSupportKind {
  if (
    citation.retrieval_kind === "semantic" &&
    citation.support_kind === "semantic_candidate" &&
    citation.support_start === citation.match_start &&
    citation.support_end === citation.match_end
  ) {
    return "semantic_candidate";
  }
  return citation.support_kind === "sentence" ? "sentence" : "retrieval_match";
}

/**
 * Return the citation's support span, falling back to the retrieval hit.
 *
 * The support fields are only trusted when they form an exact, self-consistent
 * page-text span that contains the hit. Older responses without them, or any
 * inconsistent values, use the hit span and are marked `retrieval_match`.
 */
export function citationSupportSpan(
  citation: EvidenceCitation
): CitationSupportSpan {
  const { support_text, support_start, support_end } = citation;
  const semantic = citation.retrieval_kind === "semantic";
  if (
    typeof support_text === "string" &&
    support_text.length > 0 &&
    isInteger(support_start) &&
    support_start >= 0 &&
    isInteger(support_end) &&
    support_end > support_start &&
    // Python offsets count code points; JS string length counts UTF-16 units.
    Array.from(support_text).length === support_end - support_start &&
    support_start <= citation.match_start &&
    support_end >= citation.match_end &&
    // A semantic candidate is the whole search unit: support must equal it.
    (!semantic ||
      (citation.support_kind === "semantic_candidate" &&
        support_start === citation.match_start &&
        support_end === citation.match_end))
  ) {
    return {
      text: support_text,
      start: support_start,
      end: support_end,
      kind: supportKind(citation),
    };
  }
  return {
    text: citation.hit_text,
    start: citation.match_start,
    end: citation.match_end,
    kind: semantic ? "semantic_candidate" : "retrieval_match",
  };
}

/**
 * Build one evidence/locate item from a question citation.
 *
 * The span comes straight from the server response. Clicking a citation must
 * never search `hit_text` again, and no PDF coordinates are derived here.
 */
export function toCitationEvidenceSpan(
  citation: EvidenceCitation
): EvidenceSpanRequest {
  const support = citationSupportSpan(citation);
  return {
    page_index: citation.page_index,
    match_start: support.start,
    match_end: support.end,
    hit_text: support.text,
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

/** Compact label for a support kind pill. */
export function citationSupportLabel(kind: CitationSupportKind): string {
  switch (kind) {
    case "semantic_candidate":
      return "语义候选";
    case "sentence":
      return "支持句";
    default:
      return "仅检索命中";
  }
}

/** Compact label for a citation button. */
export function citationLabel(citation: EvidenceCitation): string {
  const support = citationSupportSpan(citation);
  if (support.kind === "semantic_candidate") {
    const score =
      typeof citation.score === "number"
        ? ` · 相似度 ${citation.score.toFixed(3)}`
        : "";
    return `${citation.citation_id} · 第 ${citation.page_number} 页 · 语义候选${score} · ${support.start}–${support.end}`;
  }
  const kind = support.kind === "sentence" ? "支持句" : "仅检索命中";
  return `${citation.citation_id} · 第 ${citation.page_number} 页 · ${kind} ${support.start}–${support.end}`;
}
