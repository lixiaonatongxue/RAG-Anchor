import { describe, expect, it } from "vitest";

import {
  citationBelongsToDocument,
  citationLabel,
  citationSupportSpan,
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

const sentenceCitation: EvidenceCitation = {
  ...citation,
  match_start: 4,
  match_end: 9,
  chunk_text: "The quick brown fox jumps over the lazy dog.",
  support_text: "The quick brown fox jumps over the lazy dog.",
  support_start: 0,
  support_end: 44,
  support_kind: "sentence",
};

const fallbackCitation: EvidenceCitation = {
  ...citation,
  support_text: "quick",
  support_start: 42,
  support_end: 47,
  support_kind: "retrieval_match",
};

describe("toCitationEvidenceSpan", () => {
  it("falls back to the retrieval hit when support fields are absent", () => {
    expect(toCitationEvidenceSpan(citation)).toEqual({
      page_index: 3,
      match_start: 42,
      match_end: 47,
      hit_text: "quick",
      chunk_id: "chunk-0002",
    });
  });

  it("locates the complete support sentence when the server provides one", () => {
    expect(toCitationEvidenceSpan(sentenceCitation)).toEqual({
      page_index: 3,
      match_start: 0,
      match_end: 44,
      hit_text: "The quick brown fox jumps over the lazy dog.",
      chunk_id: "chunk-0002",
    });
  });

  it("keeps an explicit retrieval-only support span unchanged", () => {
    expect(toCitationEvidenceSpan(fallbackCitation)).toEqual({
      page_index: 3,
      match_start: 42,
      match_end: 47,
      hit_text: "quick",
      chunk_id: "chunk-0002",
    });
  });

  it("never derives PDF geometry or re-searches the hit text", () => {
    for (const item of [citation, sentenceCitation, fallbackCitation]) {
      const span = toCitationEvidenceSpan(item);
      expect(span).not.toHaveProperty("rects");
      expect(span).not.toHaveProperty("bbox");
      expect(span).not.toHaveProperty("page");
      expect(span).not.toHaveProperty("char_start");
    }
  });
});

describe("citationSupportSpan", () => {
  it("rejects a support span whose text length does not match its offsets", () => {
    const inconsistent: EvidenceCitation = {
      ...sentenceCitation,
      support_end: 45,
    };
    expect(citationSupportSpan(inconsistent)).toEqual({
      text: "quick",
      start: 4,
      end: 9,
      kind: "retrieval_match",
    });
  });

  it("rejects a support span that does not contain the retrieval hit", () => {
    const inconsistent: EvidenceCitation = {
      ...sentenceCitation,
      support_start: 10,
      support_end: 30,
      support_text: "brown fox jumps over the l",
    };
    expect(citationSupportSpan(inconsistent).kind).toBe("retrieval_match");
    expect(citationSupportSpan(inconsistent).start).toBe(4);
  });

  it("does not treat missing support data as a sentence", () => {
    expect(citationSupportSpan(citation)).toEqual({
      text: "quick",
      start: 42,
      end: 47,
      kind: "retrieval_match",
    });
  });

  it("counts code points so non-BMP text keeps Python-offset parity", () => {
    const nonBmp: EvidenceCitation = {
      ...citation,
      match_start: 2,
      match_end: 7,
      support_text: "😀 quick.",
      support_start: 0,
      support_end: 8,
      support_kind: "sentence",
    };
    expect(citationSupportSpan(nonBmp)).toEqual({
      text: "😀 quick.",
      start: 0,
      end: 8,
      kind: "sentence",
    });
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
  it("labels an older citation as retrieval-only", () => {
    expect(citationLabel(citation)).toBe("E2 · 第 4 页 · 仅检索命中 42–47");
  });

  it("labels a sentence support span and its exact offsets", () => {
    expect(citationLabel(sentenceCitation)).toBe("E2 · 第 4 页 · 支持句 0–44");
  });

  it("labels an explicit fallback as retrieval-only", () => {
    expect(citationLabel(fallbackCitation)).toBe(
      "E2 · 第 4 页 · 仅检索命中 42–47"
    );
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
