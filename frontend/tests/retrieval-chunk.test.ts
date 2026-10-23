import { describe, expect, it } from "vitest";

import {
  CHUNK_PAGE_SIZE,
  chunkEmptyMessage,
  computeChunkPage,
  evidenceErrorMessage,
  toEvidenceSpan,
} from "../src/retrieval/chunk";
import type { RetrievalHit } from "../src/api/types";

const hit: RetrievalHit = {
  document_id: "11111111-1111-1111-1111-111111111111",
  chunk_id: "chunk-0001",
  page_index: 0,
  page_number: 1,
  chunk_index: 0,
  start_offset: 10,
  end_offset: 30,
  match_start: 12,
  match_end: 17,
  hit_text: "quick",
  chunk_text: "The quick brown fox jumps over the lazy dog.",
  extraction_version: "v1",
};

describe("toEvidenceSpan", () => {
  it("passes all five request fields through unchanged", () => {
    expect(toEvidenceSpan(hit)).toEqual({
      page_index: hit.page_index,
      match_start: hit.match_start,
      match_end: hit.match_end,
      hit_text: hit.hit_text,
      chunk_id: hit.chunk_id,
    });
  });
  it("keeps a null chunk_id as null", () => {
    const span = toEvidenceSpan({
      ...hit,
      chunk_id: null,
    } as unknown as RetrievalHit);
    expect(span.chunk_id).toBeNull();
  });

  it("never invents PDF coordinates or Python offsets", () => {
    const span = toEvidenceSpan(hit);
    expect(span).not.toHaveProperty("char_start");
    expect(span).not.toHaveProperty("rects");
    expect(span).not.toHaveProperty("page");
  });
});

describe("computeChunkPage", () => {
  it("reports an empty first page for zero results", () => {
    const page = computeChunkPage(0, 0, 0);
    expect(page.hasPrevious).toBe(false);
    expect(page.hasNext).toBe(false);
    expect(page.previousOffset).toBe(0);
    expect(page.nextOffset).toBe(CHUNK_PAGE_SIZE);
    expect(page.label).toBe("0 / 0");
  });

  it("recognizes an exact full page with no next page", () => {
    const page = computeChunkPage(10, 0, 10);
    expect(page.hasPrevious).toBe(false);
    expect(page.hasNext).toBe(false);
    expect(page.label).toBe("1–10 / 10");
  });

  it("shows a middle page with both neighbours", () => {
    const page = computeChunkPage(25, 10, 10);
    expect(page.hasPrevious).toBe(true);
    expect(page.hasNext).toBe(true);
    expect(page.previousOffset).toBe(0);
    expect(page.nextOffset).toBe(20);
    expect(page.label).toBe("11–20 / 25");
  });

  it("handles the last partial page", () => {
    const page = computeChunkPage(25, 20, 5);
    expect(page.hasPrevious).toBe(true);
    expect(page.hasNext).toBe(false);
    expect(page.label).toBe("21–25 / 25");
  });

  it("treats negative or fractional inputs as empty state", () => {
    const page = computeChunkPage(-3, -2, -1);
    expect(page.hasPrevious).toBe(false);
    expect(page.hasNext).toBe(false);
    expect(page.label).toBe("0 / 0");
  });
});

describe("chunkEmptyMessage", () => {
  it("reports the searching state first", () => {
    expect(chunkEmptyMessage({ searching: true, hasSearched: false })).toBe(
      "正在检索 Chunk…"
    );
  });

  it("reports a completed search without hits", () => {
    expect(chunkEmptyMessage({ searching: false, hasSearched: true })).toBe(
      "没有找到匹配的 Chunk。"
    );
  });

  it("gives the idle instruction before any search", () => {
    expect(chunkEmptyMessage({ searching: false, hasSearched: false })).toBe(
      "检索后，命中的 Chunk 会按页码列在这里。"
    );
  });
});

describe("evidenceErrorMessage", () => {
  it("uses the exact 409 message and never falls back", () => {
    expect(evidenceErrorMessage(409, "Document has not been indexed")).toBe(
      "这条证据暂时无法可靠定位。"
    );
    expect(evidenceErrorMessage(409, "连接失败")).toBe(
      "这条证据暂时无法可靠定位。"
    );
  });

  it("passes any other status through to the fallback message", () => {
    expect(evidenceErrorMessage(404, "Document not found")).toBe(
      "Document not found"
    );
    expect(evidenceErrorMessage(0, "无法连接后端服务")).toBe(
      "无法连接后端服务"
    );
  });
});
