/**
 * TypeScript mirrors of the FastAPI response models.
 *
 * These types intentionally follow the backend contract exactly. Do not add
 * fields that the API does not return, and never reinterpret the rectangle
 * coordinates: they are absolute PDF user-space values in points (ADR-0001).
 */

export interface DocumentSummary {
  document_id: string;
  filename: string;
  size_bytes: number;
  page_count: number;
  created_at: string;
}

export interface DocumentListResponse {
  items: DocumentSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface RectData {
  left: number;
  bottom: number;
  right: number;
  top: number;
}

export interface PageData {
  page_index: number;
  width_pt: number;
  height_pt: number;
  rotation: number;
  cropbox: [number, number, number, number];
}

export interface AnchorHit {
  page_index: number;
  page_number: number;
  matched_text: string;
  char_start: number;
  char_count: number;
  rects: RectData[];
  bbox: RectData | null;
  page: PageData;
}

export interface AnchorSearchResponse {
  document_id: string;
  query: string;
  match_case: boolean;
  total: number;
  results: AnchorHit[];
}

/** Statistics returned by one finished index build. */
export interface DocumentIndexResponse {
  document_id: string;
  fingerprint: string;
  page_count: number;
  chunk_count: number;
  filtered_chunk_count: number;
  indexed_at: string;
}

/**
 * One literal match inside a persisted chunk.
 *
 * `match_start` / `match_end` are Python string offsets into the stored page
 * text. They identify the occurrence and are passed back to evidence/locate
 * unchanged; they are never PDF coordinates or PDFium character indices.
 */
export interface RetrievalHit {
  document_id: string;
  chunk_id: string;
  page_index: number;
  page_number: number;
  chunk_index: number;
  start_offset: number;
  end_offset: number;
  match_start: number;
  match_end: number;
  hit_text: string;
  chunk_text: string;
  extraction_version: string;
}

export interface ChunkRetrievalResponse {
  query: string;
  match_case: boolean;
  document_id: string | null;
  total: number;
  limit: number;
  offset: number;
  results: RetrievalHit[];
}

/** One exact page-text span whose real PDF geometry is requested. */
export interface EvidenceSpanRequest {
  page_index: number;
  match_start: number;
  match_end: number;
  hit_text: string;
  chunk_id?: string | null;
}

/** One span located in the rendered PDF with real PDFium rectangles. */
export interface LocatedEvidence {
  page_index: number;
  page_number: number;
  match_start: number;
  match_end: number;
  hit_text: string;
  chunk_id: string | null;
  char_start: number;
  char_count: number;
  matched_text: string;
  rects: RectData[];
  bbox: RectData | null;
  page: PageData;
}

export interface EvidenceLocateResponse {
  document_id: string;
  total: number;
  results: LocatedEvidence[];
}

/**
 * One retrieved page-text span offered as a question citation.
 *
 * These are Python string offsets into the stored page text, never PDF
 * coordinates. A citation is only displayable after evidence/locate maps it
 * to real PDFium geometry.
 */
export interface EvidenceCitation {
  citation_id: string;
  document_id: string;
  chunk_id: string;
  page_index: number;
  page_number: number;
  chunk_index: number;
  match_start: number;
  match_end: number;
  hit_text: string;
  chunk_text: string;
  extraction_version: string;
}

/**
 * Evidence-only question answer.
 *
 * `answer` is a fixed server notice, not model output. `citations` contains
 * only spans retrieved from the requested document.
 */
export interface QuestionAnswerResponse {
  document_id: string;
  question: string;
  retrieval_query: string;
  match_case: boolean;
  answer_mode: "evidence_only";
  status: "evidence_found" | "insufficient_evidence";
  answer: string;
  total_matches: number;
  citations: EvidenceCitation[];
}
