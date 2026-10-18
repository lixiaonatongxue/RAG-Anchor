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
