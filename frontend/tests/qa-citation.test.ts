import { describe, expect, it } from "vitest";

import {
  citationBelongsToDocument,
  citationLabel,
  toCitationEvidenceSpan,
} from "../src/qa/citation";
import type { EvidenceCitation } from "../src/api/types";
import { evidenceErrorMessage } from "../src/retrieval/chunk";
import { createRequestGate } from "../src/ui/request";

const citation: EvidenceCitation = {
  citation_id: "E2",
  document_id: "11111111-1111-1111-1111-111111111111",
  chunk_id: "chunk-0002",
  page_index: 3,
  page_number: 4,
  chunk_index: 1,
  match_start: 42,
  match_end: 47,
  hit_text: "quick",
  chunk_text: "The quick brown fox jumps over the lazy dog.",
  extraction_version: "v1",
};

describe("toCitationEvidenceSpan", () => {
  it("passes the retrieved occurrence through unchanged", () => {
    expect(toCitationEvidenceSpan(citation)).toEqual({
      page_index: 3,
      match_start: 42,
      match_end: 47,
      hit_text: "quick",
      chunk_id: "chunk-0002",
    });
  });

  it("never derives PDF geometry or re-searches the hit text", () => {
    const span = toCitationEvidenceSpan(citation);
    expect(span).not.toHaveProperty("rects");
    expect(span).not.toHaveProperty("bbox");
    expect(span).not.toHaveProperty("page");
    expect(span).not.toHaveProperty("char_start");
  });
});

describe("citationBelongsToDocument", () => {
  it("accepts a citation from the requested document", () => {
    expect(
      citationBelongsToDocument(citation, citation.document_id)
    ).toBe(true);
  });

  it("rejects a citation from another document", () => {
    expect(
      citationBelongsToDocument(citation, "99999999-9999-9999-9999-999999999999")
    ).toBe(false);
  });
});

describe("citationLabel", () => {
  it("shows the citation id, page and exact Python span", () => {
    expect(citationLabel(citation)).toBe("E2 · 第 4 页 · 字符 42–47");
  });
});

describe("question answer request gating", () => {
  it("ignores an older question response that resolves after a newer one", () => {
    const gate = createRequestGate();
    const applied: string[] = [];

    const first = gate.begin();
    const second = gate.begin();

    // Responses can arrive in any order; only the newest token may commit.
    const firstResponse = Promise.resolve().then(() => {
      if (gate.isCurrent(first)) {
        applied.push("old answer");
      }
    });
    const secondResponse = Promise.resolve().then(() => {
      if (gate.isCurrent(second)) {
        applied.push("new answer");
      }
    });

    return Promise.all([firstResponse, secondResponse]).then(() => {
      expect(applied).toEqual(["new answer"]);
    });
  });

  it("uses the same exact-409 message as chunk evidence", () => {
    expect(evidenceErrorMessage(409, "Stored PDF does not match the index")).toBe(
      "这条证据暂时无法可靠定位。"
    );
  });
});
