import type {
  AnchorSearchResponse,
  AnswerMode,
  ChunkRetrievalResponse,
  DocumentIndexResponse,
  DocumentListResponse,
  DocumentSummary,
  EvidenceLocateResponse,
  EvidenceSpanRequest,
  QuestionAnswerResponse,
} from "./types";

const API_BASE = "/api/v1";

/** An HTTP or transport failure raised by {@link RagAnchorApi}. */
export class ApiError extends Error {
  /** HTTP status code, or 0 when the request never reached the server. */
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

interface ErrorPayload {
  detail?: unknown;
}

function extractDetail(payload: unknown): string | null {
  if (typeof payload !== "object" || payload === null) {
    return null;
  }

  const detail = (payload as ErrorPayload).detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) => {
        if (typeof entry === "object" && entry !== null && "msg" in entry) {
          const message = (entry as { msg?: unknown }).msg;
          return typeof message === "string" ? message : null;
        }
        return null;
      })
      .filter((message): message is string => message !== null);
    return messages.length > 0 ? messages.join("；") : null;
  }

  return null;
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  let detail: string | null = null;
  try {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      detail = extractDetail(await response.json());
    }
  } catch {
    // A malformed error body must not mask the HTTP status.
  }
  return new ApiError(
    detail ?? `请求失败（HTTP ${response.status}）`,
    response.status
  );
}

async function request<T>(input: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(input, init);
  } catch (error) {
    const reason =
      error instanceof Error && error.message ? `：${error.message}` : "";
    throw new ApiError(`无法连接后端服务${reason}`, 0);
  }

  if (!response.ok) {
    throw await errorFromResponse(response);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

/** Thin client for the RAG-Anchor HTTP API. */
export class RagAnchorApi {
  listDocuments(limit = 100, offset = 0): Promise<DocumentListResponse> {
    const params = new URLSearchParams({
      limit: String(limit),
      offset: String(offset),
    });
    return request<DocumentListResponse>(
      `${API_BASE}/documents?${params.toString()}`
    );
  }

  getDocument(documentId: string): Promise<DocumentSummary> {
    return request<DocumentSummary>(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}`
    );
  }

  uploadDocument(file: File): Promise<DocumentSummary> {
    const body = new FormData();
    // The backend expects the multipart field name "file".
    body.append("file", file, file.name);
    return request<DocumentSummary>(`${API_BASE}/documents`, {
      method: "POST",
      body,
    });
  }

  searchAnchors(
    documentId: string,
    query: string,
    matchCase = true
  ): Promise<AnchorSearchResponse> {
    return request<AnchorSearchResponse>(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}/anchors/search`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, match_case: matchCase }),
      }
    );
  }

  indexDocument(documentId: string): Promise<DocumentIndexResponse> {
    return request<DocumentIndexResponse>(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}/index`,
      { method: "POST" }
    );
  }

  searchChunks(options: {
    query: string;
    documentId: string;
    matchCase?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<ChunkRetrievalResponse> {
    return request<ChunkRetrievalResponse>(`${API_BASE}/retrieval/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: options.query,
        document_id: options.documentId,
        match_case: options.matchCase ?? true,
        limit: options.limit ?? 10,
        offset: options.offset ?? 0,
      }),
    });
  }

  locateEvidence(
    documentId: string,
    items: readonly EvidenceSpanRequest[]
  ): Promise<EvidenceLocateResponse> {
    return request<EvidenceLocateResponse>(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}/evidence/locate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ items }),
      }
    );
  }

  askQuestion(
    documentId: string,
    options: {
      question: string;
      retrievalQuery?: string;
      matchCase?: boolean;
      maxCitations?: number;
      answerMode?: AnswerMode;
    }
  ): Promise<QuestionAnswerResponse> {
    const body: Record<string, unknown> = {
      question: options.question,
      answer_mode: options.answerMode ?? "evidence_only",
      match_case: options.matchCase ?? false,
      max_citations: options.maxCitations ?? 5,
    };
    if (options.retrievalQuery) {
      body.retrieval_query = options.retrievalQuery;
    }
    return request<QuestionAnswerResponse>(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}/qa`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );
  }

  async getDocumentFile(documentId: string): Promise<ArrayBuffer> {
    let response: Response;
    try {
      response = await fetch(
        `${API_BASE}/documents/${encodeURIComponent(documentId)}/file`
      );
    } catch (error) {
      const reason =
        error instanceof Error && error.message ? `：${error.message}` : "";
      throw new ApiError(`无法连接后端服务${reason}`, 0);
    }

    if (!response.ok) {
      throw await errorFromResponse(response);
    }

    return response.arrayBuffer();
  }

  async deleteDocument(documentId: string): Promise<void> {
    await request<void>(
      `${API_BASE}/documents/${encodeURIComponent(documentId)}`,
      {
        method: "DELETE",
      }
    );
  }
}
