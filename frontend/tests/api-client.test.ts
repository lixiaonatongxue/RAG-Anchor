import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, RagAnchorApi } from "../src/api/client";
import type {
  AnchorSearchResponse,
  ChunkRetrievalResponse,
  DocumentIndexResponse,
  DocumentListResponse,
  DocumentSummary,
  EvidenceLocateResponse,
  EvidenceSpanRequest,
  LocatedEvidence,
  RetrievalHit,
} from "../src/api/types";

const api = new RagAnchorApi();

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function makeFile(name = "evidence.pdf"): File {
  const blob = new Blob(["%PDF-1.7"], { type: "application/pdf" });
  return Object.assign(blob, { name }) as unknown as File;
}

const documentSummary: DocumentSummary = {
  document_id: "11111111-1111-1111-1111-111111111111",
  filename: "evidence.pdf",
  size_bytes: 1024,
  page_count: 3,
  created_at: "2026-10-18T10:00:00Z",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("RagAnchorApi.listDocuments", () => {
  it("requests the document list with pagination and parses the response", async () => {
    const payload: DocumentListResponse = {
      items: [documentSummary],
      total: 1,
      limit: 100,
      offset: 0,
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.listDocuments(100, 0);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe(
      "/api/v1/documents?limit=100&offset=0"
    );
    expect(result).toEqual(payload);
  });
});

describe("RagAnchorApi.uploadDocument", () => {
  it("posts multipart form data using the backend field name 'file'", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(documentSummary, 201));
    vi.stubGlobal("fetch", fetchMock);

    const file = makeFile("my evidence.pdf");
    const result = await api.uploadDocument(file);

    expect(result).toEqual(documentSummary);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/documents");
    expect(init?.method).toBe("POST");

    const body = init?.body;
    expect(body).toBeInstanceOf(FormData);
    const form = body as FormData;
    expect(form.has("file")).toBe(true);
    expect(form.get("file")).not.toBeNull();
  });
});

describe("RagAnchorApi.searchAnchors", () => {
  it("sends query and match_case as JSON and parses hits", async () => {
    const payload: AnchorSearchResponse = {
      document_id: documentSummary.document_id,
      query: "quick",
      match_case: true,
      total: 1,
      results: [
        {
          page_index: 0,
          page_number: 1,
          matched_text: "quick",
          char_start: 4,
          char_count: 5,
          rects: [
            { left: 124.432, bottom: 197.528, right: 152.032, top: 208.616 },
          ],
          bbox: {
            left: 124.432,
            bottom: 197.528,
            right: 152.032,
            top: 208.616,
          },
          page: {
            page_index: 0,
            width_pt: 612,
            height_pt: 792,
            rotation: 0,
            cropbox: [0, 0, 612, 792],
          },
        },
      ],
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.searchAnchors(
      documentSummary.document_id,
      "quick",
      true
    );

    expect(result).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      `/api/v1/documents/${documentSummary.document_id}/anchors/search`
    );
    expect(init?.method).toBe("POST");
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(String(init?.body))).toEqual({
      query: "quick",
      match_case: true,
    });
  });

  it("returns an empty result set for zero hits", async () => {
    const payload: AnchorSearchResponse = {
      document_id: documentSummary.document_id,
      query: "zzzznotfound",
      match_case: true,
      total: 0,
      results: [],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(payload))
    );

    const result = await api.searchAnchors(
      documentSummary.document_id,
      "zzzznotfound"
    );

    expect(result.total).toBe(0);
    expect(result.results).toEqual([]);
  });
});

describe("RagAnchorApi error handling", () => {
  it("maps an HTTP error body's detail into ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(jsonResponse({ detail: "Document not found" }, 404))
    );

    await expect(
      api.getDocument(documentSummary.document_id)
    ).rejects.toMatchObject({
      name: "ApiError",
      message: "Document not found",
      status: 404,
    });
  });

  it("joins FastAPI validation errors into a readable message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        jsonResponse(
          {
            detail: [
              {
                loc: ["body", "query"],
                msg: "field required",
                type: "missing",
              },
              {
                loc: ["body", "match_case"],
                msg: "invalid",
                type: "type_error",
              },
            ],
          },
          422
        )
      )
    );

    await expect(
      api.searchAnchors(documentSummary.document_id, "quick")
    ).rejects.toThrow("field required；invalid");
  });

  it("wraps transport failures in ApiError with status 0", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockRejectedValue(new TypeError("Failed to fetch"))
    );

    await expect(api.listDocuments()).rejects.toMatchObject({
      name: "ApiError",
      status: 0,
    });
  });

  it("exposes ApiError as an Error subclass", () => {
    const error = new ApiError("boom", 500);
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe("boom");
    expect(error.status).toBe(500);
  });
});

describe("RagAnchorApi.getDocumentFile", () => {
  it("returns the PDF bytes as an ArrayBuffer", async () => {
    const bytes = new Uint8Array([0x25, 0x50, 0x44, 0x46]);
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(bytes, {
        status: 200,
        headers: { "Content-Type": "application/pdf" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const buffer = await api.getDocumentFile(documentSummary.document_id);

    expect(fetchMock.mock.calls[0][0]).toBe(
      `/api/v1/documents/${documentSummary.document_id}/file`
    );
    expect(new Uint8Array(buffer)).toEqual(bytes);
  });
});

describe("RagAnchorApi.deleteDocument", () => {
  it("accepts a 204 response with no body", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(null, { status: 204 }))
    );

    await expect(
      api.deleteDocument(documentSummary.document_id)
    ).resolves.toBeUndefined();
  });
});

const indexResponse: DocumentIndexResponse = {
  document_id: "11111111-1111-1111-1111-111111111111",
  fingerprint: "abc123",
  page_count: 3,
  chunk_count: 25,
  filtered_chunk_count: 22,
  indexed_at: "2026-10-18T10:00:00Z",
};

const retrievalHit: RetrievalHit = {
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

const locatedEvidence: LocatedEvidence = {
  page_index: 3,
  page_number: 4,
  match_start: 12,
  match_end: 17,
  hit_text: "quick",
  chunk_id: "chunk-0001",
  char_start: 40,
  char_count: 5,
  matched_text: "quick",
  rects: [{ left: 124.432, bottom: 197.528, right: 152.032, top: 208.616 }],
  bbox: { left: 124.432, bottom: 197.528, right: 152.032, top: 208.616 },
  page: {
    page_index: 3,
    width_pt: 612,
    height_pt: 792,
    rotation: 0,
    cropbox: [0, 0, 612, 792],
  },
};

describe("RagAnchorApi.indexDocument", () => {
  it("posts to the index endpoint with no body and parses stats", async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(indexResponse));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.indexDocument(documentSummary.document_id);
    expect(result).toEqual(indexResponse);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(`/api/v1/documents/${documentSummary.document_id}/index`);
    expect(init?.method).toBe("POST");
    expect(init?.body).toBeUndefined();
    expect(init?.headers).toBeUndefined();
  });
});

describe("RagAnchorApi.searchChunks", () => {
  it("sends query, document_id, match_case, limit and offset as JSON", async () => {
    const payload: ChunkRetrievalResponse = {
      query: "quick",
      match_case: true,
      document_id: documentSummary.document_id,
      total: 2,
      limit: 10,
      offset: 0,
      results: [retrievalHit],
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.searchChunks({
      query: "quick",
      documentId: documentSummary.document_id,
      matchCase: true,
      limit: 10,
      offset: 0,
    });

    expect(result).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/v1/retrieval/search");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({
      query: "quick",
      document_id: documentSummary.document_id,
      match_case: true,
      limit: 10,
      offset: 0,
    });
  });

  it("defaults match_case to true and pagination to 10 / 0", async () => {
    const payload: ChunkRetrievalResponse = {
      query: "brown",
      match_case: true,
      document_id: documentSummary.document_id,
      total: 0,
      limit: 10,
      offset: 0,
      results: [],
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    await api.searchChunks({
      query: "brown",
      documentId: documentSummary.document_id,
    });

    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toEqual({
      query: "brown",
      document_id: documentSummary.document_id,
      match_case: true,
      limit: 10,
      offset: 0,
    });
  });
});

describe("RagAnchorApi.locateEvidence", () => {
  it("posts the items array unchanged and parses located results", async () => {
    const items: EvidenceSpanRequest[] = [
      {
        page_index: 0,
        match_start: 12,
        match_end: 17,
        hit_text: "quick",
        chunk_id: "chunk-0001",
      },
    ];
    const payload: EvidenceLocateResponse = {
      document_id: documentSummary.document_id,
      total: 1,
      results: [locatedEvidence],
    };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse(payload));
    vi.stubGlobal("fetch", fetchMock);

    const result = await api.locateEvidence(documentSummary.document_id, items);

    expect(result).toEqual(payload);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      `/api/v1/documents/${documentSummary.document_id}/evidence/locate`
    );
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ items });
  });
});
