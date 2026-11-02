import type { EvidenceSpanRequest, RetrievalHit } from "../api/types";

/** Server-side page size for chunk retrieval. */
export const CHUNK_PAGE_SIZE = 10;

/**
 * Build one evidence/locate item from a retrieval hit.
 *
 * The offsets come straight from the server response. Nothing here searches
 * `chunk_text` again or converts `chunk_index` into a page number: the backend
 * already identified the exact occurrence inside the stored page text.
 */
export function toEvidenceSpan(hit: RetrievalHit): EvidenceSpanRequest {
  return {
    page_index: hit.page_index,
    match_start: hit.match_start,
    match_end: hit.match_end,
    hit_text: hit.hit_text,
    chunk_id: hit.chunk_id,
  };
}

export interface ChunkPage {
  hasPrevious: boolean;
  hasNext: boolean;
  previousOffset: number;
  nextOffset: number;
  label: string;
}

/** Derive pagination state from the server total, never from `chunk_index`. */
export function computeChunkPage(
  total: number,
  offset: number,
  count: number,
  pageSize = CHUNK_PAGE_SIZE,
): ChunkPage {
  const safeTotal = Math.max(0, Math.trunc(total));
  const safeOffset = Math.max(0, Math.trunc(offset));
  const safeCount = Math.max(0, Math.trunc(count));
  const hasPrevious = safeOffset > 0;
  const hasNext = safeOffset + safeCount < safeTotal;
  const first = safeCount === 0 ? 0 : safeOffset + 1;
  const last = safeOffset + safeCount;

  return {
    hasPrevious,
    hasNext,
    previousOffset: Math.max(0, safeOffset - pageSize),
    nextOffset: safeOffset + pageSize,
    label: safeCount === 0 ? `0 / ${safeTotal}` : `${first}–${last} / ${safeTotal}`,
  };
}

/** Empty-state copy for the chunk result list. */
export function chunkEmptyMessage(options: {
  searching: boolean;
  hasSearched: boolean;
}): string {
  if (options.searching) {
    return "正在检索…";
  }
  if (options.hasSearched) {
    return "没有找到匹配的 Chunk。";
  }
  return "暂无结果";
}

/**
 * A 409 means the backend could not prove exact geometry for this span.
 * Callers must show this message and must not search for the same text or
 * approximate a rectangle as a fallback.
 */
export function evidenceErrorMessage(status: number, fallback: string): string {
  if (status === 409) {
    return "这条证据暂时无法可靠定位。";
  }
  return fallback;
}
