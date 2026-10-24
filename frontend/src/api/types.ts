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
 * How a citation's support span was derived.
 *
 * `sentence` means the complete sentence containing the retrieval hit was
 * selected by a deterministic punctuation heuristic. It is not proof that the
 * sentence supports every generated conclusion. `retrieval_match` means no
 * reliable sentence could be selected, so the support span is exactly the
 * literal retrieval hit. `semantic_candidate` means the span is one ranked
 * semantic candidate; it is not a keyword occurrence.
 */
export type CitationSupportKind =
  | "sentence"
  | "retrieval_match"
  | "semantic_candidate";

/** Retrieval layer used for one question. */
export type RetrievalMode = "literal" | "semantic";

/**
 * One retrieved page-text span offered as a question citation.
 *
 * These are Python string offsets into the stored page text, never PDF
 * coordinates. A citation is only displayable after evidence/locate maps it
 * to real PDFium geometry.
 *
 * `support_*` fields are additive. When absent (an older server), callers fall
 * back to the retrieval hit span. Semantic citations carry
 * `retrieval_kind == "semantic"` and a cosine `score`.
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
  retrieval_kind?: RetrievalMode;
  score?: number | null;
  support_text?: string;
  support_start?: number;
  support_end?: number;
  support_kind?: CitationSupportKind;
}

/**
 * Question-answering modes.
 *
 * `evidence_only` never calls a model. `grounded_generation` sends retrieved
 * evidence to a configured model and validates its citation IDs server-side.
 */
export type AnswerMode = "evidence_only" | "grounded_generation";

export type QuestionStatus =
  | "evidence_found"
  | "insufficient_evidence"
  | "answered";

/**
 * One question answer.
 *
 * In `evidence_only` mode `answer` is a fixed server notice. In
 * `grounded_generation` mode it is model text whose citation IDs were
 * validated by the server; that does not prove every sentence is supported.
 * `citations` always contains server-owned spans from the requested document.
 */
export interface QuestionAnswerResponse {
  document_id: string;
  question: string;
  retrieval_query: string;
  retrieval_mode: RetrievalMode;
  match_case: boolean;
  answer_mode: AnswerMode;
  status: QuestionStatus;
  answer: string;
  total_matches: number;
  citations: EvidenceCitation[];
}

/** Provenance of one derived semantic index build. */
export interface SemanticIndexResponse {
  document_id: string;
  model_name: string;
  model_version: string;
  model_digest: string;
  index_version: number;
  candidate_version: number;
  dimension: number;
  normalization: "l2";
  source_fingerprint: string;
  candidate_count: number;
  built_at: string;
}
