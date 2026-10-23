import { ApiError, RagAnchorApi } from "./api/client";
import type {
  AnchorHit,
  DocumentIndexResponse,
  DocumentSummary,
  RetrievalHit,
} from "./api/types";
import { paintHighlights } from "./highlight/renderer";
import {
  anchorSelection,
  emptySelection,
  evidenceSelection,
  selectPageHighlights,
} from "./highlight/selection";
import type { HighlightSelection } from "./highlight/selection";
import { PdfViewer } from "./pdf/viewer";
import type { ViewerState } from "./pdf/viewer";
import {
  CHUNK_PAGE_SIZE,
  chunkEmptyMessage,
  computeChunkPage,
  evidenceErrorMessage,
  toEvidenceSpan,
} from "./retrieval/chunk";
import { createRequestGate } from "./ui/request";
import {
  centerTargetsWhenReady,
  createScrollRequestGate,
  revealWithin,
  waitForLayout,
} from "./ui/scroll";
import type { PageViewport } from "pdfjs-dist";
import "./styles.css";

const api = new RagAnchorApi();

function byId<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`页面缺少必需的元素 #${id}`);
  }
  return element as T;
}

const connectionStatus = byId<HTMLDivElement>("connection-status");
const connectionLabel = byId<HTMLSpanElement>("connection-label");

const uploadZone = byId<HTMLLabelElement>("upload-zone");
const uploadInput = byId<HTMLInputElement>("upload-input");
const uploadStatus = byId<HTMLParagraphElement>("upload-status");

const documentCount = byId<HTMLSpanElement>("document-count");
const documentList = byId<HTMLDivElement>("document-list");

const indexStatus = byId<HTMLSpanElement>("index-status");
const indexButton = byId<HTMLButtonElement>("index-button");
const indexDetail = byId<HTMLSpanElement>("index-detail");
const chunkForm = byId<HTMLFormElement>("chunk-form");
const chunkInput = byId<HTMLTextAreaElement>("chunk-input");
const chunkMatchCase = byId<HTMLInputElement>("chunk-match-case");
const chunkSearchButton = byId<HTMLButtonElement>("chunk-search-button");
const chunkHint = byId<HTMLParagraphElement>("chunk-hint");
const chunkPosition = byId<HTMLSpanElement>("chunk-position");
const chunkPrev = byId<HTMLButtonElement>("chunk-prev");
const chunkNext = byId<HTMLButtonElement>("chunk-next");
const chunkResults = byId<HTMLDivElement>("chunk-results");

const searchForm = byId<HTMLFormElement>("search-form");
const searchInput = byId<HTMLTextAreaElement>("search-input");
const matchCase = byId<HTMLInputElement>("match-case");
const searchButton = byId<HTMLButtonElement>("search-button");
const searchHint = byId<HTMLParagraphElement>("search-hint");

const resultCount = byId<HTMLSpanElement>("result-count");
const hitPosition = byId<HTMLSpanElement>("hit-position");
const hitPrev = byId<HTMLButtonElement>("hit-prev");
const hitNext = byId<HTMLButtonElement>("hit-next");
const resultsList = byId<HTMLDivElement>("results-list");

const viewerTitle = byId<HTMLHeadingElement>("viewer-title");
const viewerSubtitle = byId<HTMLParagraphElement>("viewer-subtitle");
const prevPage = byId<HTMLButtonElement>("prev-page");
const nextPage = byId<HTMLButtonElement>("next-page");
const pageInput = byId<HTMLInputElement>("page-input");
const pageTotal = byId<HTMLSpanElement>("page-total");
const zoomOut = byId<HTMLButtonElement>("zoom-out");
const zoomIn = byId<HTMLButtonElement>("zoom-in");
const zoomLabel = byId<HTMLSpanElement>("zoom-label");
const fitWidth = byId<HTMLButtonElement>("fit-width");
const viewerStage = byId<HTMLDivElement>("viewer-stage");
const viewerEmpty = byId<HTMLDivElement>("viewer-empty");
const viewerLoading = byId<HTMLDivElement>("viewer-loading");
const viewerError = byId<HTMLDivElement>("viewer-error");
const viewerErrorMessage = byId<HTMLParagraphElement>("viewer-error-message");
const pageShell = byId<HTMLDivElement>("page-shell");
const pdfCanvas = byId<HTMLCanvasElement>("pdf-canvas");
const highlightLayer = byId<HTMLDivElement>("highlight-layer");
const toastRegion = byId<HTMLDivElement>("toast-region");

const DOCUMENT_ICON = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
    <path d="M7 3h7l4 4v14H7V3Z" stroke-linejoin="round" />
    <path d="M14 3v4h4" stroke-linejoin="round" />
    <path d="M10 12h5M10 16h5" stroke-linecap="round" />
  </svg>`;

const TRASH_ICON = `
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7">
    <path d="M5 7h14" stroke-linecap="round" />
    <path d="M9 7V5h6v2" stroke-linejoin="round" />
    <path d="M7 7l1 13h8l1-13" stroke-linejoin="round" />
    <path d="M10 11v5M14 11v5" stroke-linecap="round" />
  </svg>`;

let documents: DocumentSummary[] = [];
let totalDocuments = 0;
let selectedId: string | null = null;
let hits: AnchorHit[] = [];
let activeHitIndex = -1;
/**
 * The single highlight input for the overlay. Anchor search and chunk evidence
 * both publish into this value, so there is exactly one coordinate conversion
 * path (`highlight/renderer.ts`). A newer selection always replaces the old
 * one instead of being drawn next to it.
 */
let highlightSelection: HighlightSelection = emptySelection();
let hasSearched = false;
let searching = false;
let searchInFlight = false;
let uploadInFlight = false;
let selectionToken = 0;
let searchToken = 0;
const scrollGate = createScrollRequestGate();

/* ---------- Index and chunk retrieval state ---------- */

let indexedDocumentId: string | null = null;
let indexStats: DocumentIndexResponse | null = null;
let indexInFlight = false;

let chunkHits: RetrievalHit[] = [];
let chunkTotal = 0;
let chunkOffset = 0;
let chunkActiveIndex = -1;
let chunkHasSearched = false;
let chunkSearching = false;
let chunkSearchInFlight = false;
const chunkGate = createRequestGate();
const locateGate = createRequestGate();

const viewer = new PdfViewer(
  {
    stage: viewerStage,
    shell: pageShell,
    canvas: pdfCanvas,
    overlay: highlightLayer,
  },
  {
    onStateChange: handleViewerState,
    onRender: handleRender,
    onError: (error) => {
      showViewerError(friendlyError(error));
      showToast(friendlyError(error), "error");
    },
  }
);

/* ---------- Formatting ---------- */

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "—";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function snippet(text: string, maxLength = 150): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length > maxLength
    ? `${normalized.slice(0, maxLength)}…`
    : normalized;
}

function friendlyError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 0) {
      return "无法连接后端服务，请确认后端已在 8000 端口启动。";
    }
    if (error.status === 413) {
      return "文件超过 20 MB 上传上限。";
    }
    switch (error.message) {
      case "Document not found":
        return "文档不存在，可能已被删除。";
      case "Stored PDF is missing or unreadable":
        return "服务器上的 PDF 文件缺失或无法读取。";
      case "Stored PDF could not be searched":
        return "PDF 搜索失败，文件可能已损坏。";
      default:
        return error.message;
    }
  }
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return "发生未知错误，请稍后重试。";
}

function showToast(
  message: string,
  kind: "info" | "success" | "error" = "info"
): void {
  const toast = document.createElement("div");
  toast.className = `toast toast--${kind}`;
  toast.textContent = message;
  toastRegion.append(toast);
  window.setTimeout(() => {
    toast.classList.add("is-leaving");
    window.setTimeout(() => toast.remove(), 200);
  }, 4200);
}

function setConnection(state: "idle" | "ok" | "error", label: string): void {
  connectionStatus.dataset.state = state;
  connectionLabel.textContent = label;
}

/* ---------- Viewer integration ---------- */

function handleViewerState(state: ViewerState): void {
  pageInput.value = String(state.pageNumber);
  pageInput.disabled = !state.loaded;
  pageTotal.textContent = String(state.pageCount);
  prevPage.disabled = !state.loaded || state.pageNumber <= 1;
  nextPage.disabled = !state.loaded || state.pageNumber >= state.pageCount;
  zoomIn.disabled = !state.loaded;
  zoomOut.disabled = !state.loaded;
  fitWidth.disabled = !state.loaded;
  fitWidth.classList.toggle("is-active", state.loaded && state.fitWidth);
  zoomLabel.textContent = `${Math.round(state.scale * 100)}%`;
  pageShell.hidden = !state.loaded;
  if (state.loaded) {
    viewerEmpty.hidden = true;
  }
}

function handleRender(pageNumber: number, viewport: PageViewport): void {
  // Switching pages must not leave a highlight from another page on the
  // overlay, and a zoom re-render must map the stored rectangles again.
  const page = selectPageHighlights(highlightSelection, pageNumber);
  paintHighlights(highlightLayer, page.hits, viewport, page.activeLocalIndex);
}

function showViewerError(message: string): void {
  viewerLoading.hidden = true;
  viewerEmpty.hidden = true;
  pageShell.hidden = true;
  viewerError.hidden = false;
  viewerErrorMessage.textContent = message;
}

/* ---------- Document list ---------- */

function renderDocuments(): void {
  documentList.replaceChildren();

  if (documents.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list-empty";
    empty.textContent = "还没有文档。上传一个 PDF 开始探索。";
    documentList.append(empty);
    return;
  }

  for (const item of documents) {
    const card = document.createElement("div");
    const isActive = item.document_id === selectedId;
    card.className = `document-card${isActive ? " is-active" : ""}`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-pressed", String(isActive));
    card.title = item.filename;

    const icon = document.createElement("span");
    icon.className = "document-icon";
    icon.innerHTML = DOCUMENT_ICON;
    icon.setAttribute("aria-hidden", "true");

    const body = document.createElement("div");
    body.className = "document-body";

    const name = document.createElement("p");
    name.className = "document-name";
    name.textContent = item.filename;

    const meta = document.createElement("p");
    meta.className = "document-meta";
    meta.textContent = `${item.page_count} 页 · ${formatBytes(
      item.size_bytes
    )}`;

    const date = document.createElement("p");
    date.className = "document-date";
    date.textContent = formatDate(item.created_at);

    body.append(name, meta, date);

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "document-delete";
    removeButton.title = `删除 ${item.filename}`;
    removeButton.setAttribute("aria-label", `删除 ${item.filename}`);
    removeButton.innerHTML = TRASH_ICON;
    removeButton.addEventListener("click", (event) => {
      event.stopPropagation();
      void deleteDocument(item);
    });

    card.append(icon, body, removeButton);
    card.addEventListener("click", () => {
      void selectDocument(item.document_id);
    });
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        void selectDocument(item.document_id);
      }
    });

    documentList.append(card);
  }
}

function renderDocumentError(message: string): void {
  documentList.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "list-empty is-error";
  empty.textContent = message;
  documentList.append(empty);
}

async function loadDocuments(preferredId?: string): Promise<void> {
  try {
    const response = await api.listDocuments(100, 0);
    documents = response.items;
    totalDocuments = response.total;
    documentCount.textContent = String(totalDocuments);
    setConnection("ok", "后端已连接");
    renderDocuments();

    if (
      preferredId &&
      documents.some((item) => item.document_id === preferredId)
    ) {
      await selectDocument(preferredId);
      return;
    }

    if (
      selectedId &&
      !documents.some((item) => item.document_id === selectedId)
    ) {
      await clearSelection();
    }
  } catch (error) {
    const message = friendlyError(error);
    documents = [];
    totalDocuments = 0;
    documentCount.textContent = "0";
    setConnection("error", "后端连接失败");
    renderDocumentError(message);
  }
}

async function selectDocument(documentId: string): Promise<void> {
  const document = documents.find((item) => item.document_id === documentId);
  if (!document) {
    return;
  }

  const token = ++selectionToken;
  selectedId = documentId;
  renderDocuments();
  resetSearchResults();
  resetChunkState();
  resetIndexState();
  updateSearchAvailability();

  viewerTitle.textContent = document.filename;
  viewerSubtitle.textContent = `${document.page_count} 页 · ${formatBytes(
    document.size_bytes
  )} · ${formatDate(document.created_at)}`;
  viewerEmpty.hidden = true;
  viewerError.hidden = true;
  viewerLoading.hidden = false;
  pageShell.hidden = true;
  highlightLayer.replaceChildren();

  try {
    // Refresh the metadata from the detail endpoint before loading the file.
    const detail = await api.getDocument(documentId);
    if (token !== selectionToken) {
      return;
    }

    viewerTitle.textContent = detail.filename;
    viewerSubtitle.textContent = `${detail.page_count} 页 · ${formatBytes(
      detail.size_bytes
    )} · ${formatDate(detail.created_at)}`;

    const buffer = await api.getDocumentFile(documentId);
    if (token !== selectionToken) {
      return;
    }
    await viewer.open(buffer);
    if (token !== selectionToken) {
      return;
    }
    viewerLoading.hidden = true;
    viewerError.hidden = true;
    viewerEmpty.hidden = true;
  } catch (error) {
    if (token !== selectionToken) {
      return;
    }
    viewerLoading.hidden = true;
    showViewerError(friendlyError(error));
    showToast(friendlyError(error), "error");
    if (error instanceof ApiError && error.status === 404) {
      await loadDocuments();
    }
  }
}

async function clearSelection(): Promise<void> {
  selectionToken += 1;
  selectedId = null;
  resetSearchResults();
  resetChunkState();
  resetIndexState();
  updateSearchAvailability();
  renderDocuments();

  viewerTitle.textContent = "未选择文档";
  viewerSubtitle.textContent = "从左侧文档库选择一个 PDF 开始";
  viewerLoading.hidden = true;
  viewerError.hidden = true;
  viewerEmpty.hidden = false;
  pageShell.hidden = true;
  highlightLayer.replaceChildren();
  await viewer.close();
}

async function deleteDocument(document: DocumentSummary): Promise<void> {
  const confirmed = window.confirm(
    `确定删除「${document.filename}」吗？此操作不可恢复。`
  );
  if (!confirmed) {
    return;
  }

  try {
    await api.deleteDocument(document.document_id);
    showToast("文档已删除。", "success");
    if (selectedId === document.document_id) {
      await clearSelection();
    }
    await loadDocuments();
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

/* ---------- Upload ---------- */

async function handleUpload(file: File): Promise<void> {
  if (uploadInFlight) {
    return;
  }

  const isPdf =
    file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
  if (!isPdf) {
    uploadStatus.textContent = "请选择 PDF 文件。";
    showToast("请选择 PDF 文件。", "error");
    return;
  }

  uploadInFlight = true;
  uploadZone.classList.add("is-busy");
  uploadInput.disabled = true;
  uploadStatus.textContent = `正在上传「${file.name}」…`;

  try {
    const document = await api.uploadDocument(file);
    uploadStatus.textContent = "";
    showToast(`已上传「${document.filename}」。`, "success");
    await loadDocuments(document.document_id);
  } catch (error) {
    const message = friendlyError(error);
    uploadStatus.textContent = "上传失败。";
    showToast(message, "error");
  } finally {
    uploadInFlight = false;
    uploadZone.classList.remove("is-busy");
    uploadInput.disabled = false;
    uploadInput.value = "";
  }
}

let dragDepth = 0;

for (const eventName of ["dragenter", "dragover"] as const) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (eventName === "dragenter") {
      dragDepth += 1;
    }
    uploadZone.classList.add("is-dragover");
  });
}

uploadZone.addEventListener("dragleave", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) {
    uploadZone.classList.remove("is-dragover");
  }
});

uploadZone.addEventListener("drop", (event) => {
  event.preventDefault();
  event.stopPropagation();
  dragDepth = 0;
  uploadZone.classList.remove("is-dragover");
  const file = event.dataTransfer?.files.item(0);
  if (file) {
    void handleUpload(file);
  }
});

uploadInput.addEventListener("change", () => {
  const file = uploadInput.files?.item(0);
  if (file) {
    void handleUpload(file);
  }
});

/* ---------- Search and hit navigation ---------- */

function updateSearchButton(): void {
  searchButton.disabled =
    searchInFlight || !selectedId || searchInput.value.trim().length === 0;
}

function updateSearchAvailability(): void {
  const document = documents.find((item) => item.document_id === selectedId);

  if (!document) {
    searchHint.className = "search-hint";
    searchHint.textContent = "请先选择一个文档。";
  } else if (!hasSearched && !searchInFlight) {
    searchHint.className = "search-hint";
    searchHint.textContent = `将在「${document.filename}」中检索。`;
  }

  updateSearchButton();
  updateIndexAvailability();
  updateChunkAvailability();
}

function resetSearchResults(): void {
  hits = [];
  activeHitIndex = -1;
  highlightSelection = emptySelection();
  hasSearched = false;
  searching = false;
  searchToken += 1;
  searchInFlight = false;
  searchButton.textContent = "搜索";
  highlightLayer.replaceChildren();
  renderResults();
  updateSearchAvailability();
}

/* ---------- Index and chunk retrieval ---------- */

function resetIndexState(): void {
  indexedDocumentId = null;
  indexStats = null;
  indexInFlight = false;
  indexStatus.textContent = "未索引";
}

function resetChunkState(): void {
  chunkGate.begin();
  locateGate.begin();
  chunkHits = [];
  chunkTotal = 0;
  chunkOffset = 0;
  chunkActiveIndex = -1;
  chunkHasSearched = false;
  chunkSearching = false;
  chunkSearchInFlight = false;
  chunkPosition.textContent = "0 / 0";
  chunkHint.className = "search-hint";
  chunkHint.textContent = selectedId
    ? "尚未建立索引，先建立索引后再检索。"
    : "请先选择一个文档。";
  chunkPrev.disabled = true;
  chunkNext.disabled = true;
  chunkSearchButton.disabled = true;
  renderChunkResults();
}

function updateIndexAvailability(): void {
  const document = documents.find((item) => item.document_id === selectedId);
  if (!document) {
    indexButton.disabled = true;
    indexButton.textContent = "建立索引";
    indexDetail.textContent = "选择一个文档后可建立索引。";
    return;
  }
  if (indexInFlight) {
    indexButton.disabled = true;
    indexButton.textContent = "索引中…";
    indexDetail.textContent = "正在建立索引，请稍候。";
    return;
  }
  if (indexedDocumentId === document.document_id && indexStats) {
    indexButton.disabled = false;
    indexButton.textContent = "重新索引";
    indexDetail.textContent = `已索引 ${indexStats.page_count} 页 · ${indexStats.chunk_count} 个 Chunk。`;
    return;
  }
  indexButton.disabled = false;
  indexButton.textContent = "建立索引";
  indexDetail.textContent = "建立索引后可进行 Chunk 字面检索。";
}

function updateChunkAvailability(): void {
  const document = documents.find((item) => item.document_id === selectedId);
  const indexed = indexedDocumentId === selectedId;
  const canSearch = Boolean(document) && indexed && !chunkSearchInFlight;
  chunkSearchButton.disabled =
    !canSearch || chunkInput.value.trim().length === 0;
}

async function performIndex(): Promise<void> {
  const documentId = selectedId;
  if (!documentId || indexInFlight) {
    return;
  }
  indexInFlight = true;
  indexStatus.textContent = "索引中";
  updateIndexAvailability();

  try {
    const stats = await api.indexDocument(documentId);
    if (selectedId !== documentId) {
      return;
    }
    indexedDocumentId = documentId;
    indexStats = stats;
    indexStatus.textContent = "已索引";
    showToast(
      `已建立索引：${stats.page_count} 页 · ${stats.chunk_count} 个 Chunk。`,
      "success"
    );
  } catch (error) {
    if (selectedId !== documentId) {
      return;
    }
    indexStatus.textContent = "索引失败";
    showToast(friendlyError(error), "error");
  } finally {
    if (selectedId === documentId) {
      indexInFlight = false;
      updateIndexAvailability();
      updateChunkAvailability();
    }
  }
}

function renderChunkResults(): void {
  chunkResults.replaceChildren();

  if (chunkHits.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list-empty";
    empty.textContent = chunkEmptyMessage({
      searching: chunkSearching,
      hasSearched: chunkHasSearched,
    });
    chunkResults.append(empty);
  } else {
    chunkHits.forEach((hit, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `result-item${
        index === chunkActiveIndex ? " is-active" : ""
      }`;
      button.setAttribute(
        "aria-current",
        index === chunkActiveIndex ? "true" : "false"
      );

      const topLine = document.createElement("span");
      topLine.className = "result-topline";

      const pagePill = document.createElement("span");
      pagePill.className = "page-pill";
      pagePill.textContent = `第 ${hit.page_number} 页`;

      const charRange = document.createElement("span");
      charRange.className = "result-char";
      charRange.textContent = `字符 ${hit.match_start}–${hit.match_end}`;

      topLine.append(pagePill, charRange);

      const text = document.createElement("p");
      text.className = "result-text";
      text.textContent = snippet(hit.hit_text);

      button.append(topLine, text);
      button.addEventListener("click", () => {
        void locateChunkHit(index);
      });
      chunkResults.append(button);
    });
  }

  const page = computeChunkPage(chunkTotal, chunkOffset, chunkHits.length);
  chunkPosition.textContent = page.label;
  chunkPrev.disabled = !page.hasPrevious || chunkSearchInFlight;
  chunkNext.disabled = !page.hasNext || chunkSearchInFlight;
}

async function locateChunkHit(index: number): Promise<void> {
  const hit = chunkHits[index];
  const documentId = selectedId;
  if (!hit || !documentId) {
    return;
  }

  chunkActiveIndex = index;
  renderChunkResults();

  const requestToken = locateGate.begin();
  try {
    const response = await api.locateEvidence(documentId, [
      toEvidenceSpan(hit),
    ]);
    if (!locateGate.isCurrent(requestToken) || selectedId !== documentId) {
      return;
    }
    const located = response.results[0];
    if (!located) {
      chunkHint.className = "search-hint is-error";
      chunkHint.textContent = "服务端没有返回定位结果。";
      return;
    }

    highlightSelection = evidenceSelection(located);
    await viewer.setPage(located.page_number);

    const scrollToken = scrollGate.begin();
    await centerActiveHighlight(scrollToken);
  } catch (error) {
    if (!locateGate.isCurrent(requestToken)) {
      return;
    }
    const status = error instanceof ApiError ? error.status : 0;
    const message = evidenceErrorMessage(status, friendlyError(error));
    chunkHint.className = "search-hint is-error";
    chunkHint.textContent = message;
    showToast(message, "error");
  }
}

async function runChunkSearch(offset: number): Promise<void> {
  const documentId = selectedId;
  if (!documentId || chunkSearchInFlight) {
    return;
  }
  if (indexedDocumentId !== documentId) {
    chunkHint.className = "search-hint is-error";
    chunkHint.textContent = "请先建立索引。";
    return;
  }
  const query = chunkInput.value.trim();
  if (!query) {
    chunkHint.className = "search-hint is-error";
    chunkHint.textContent = "请输入要检索的文字。";
    chunkInput.focus();
    return;
  }

  const token = chunkGate.begin();
  chunkSearchInFlight = true;
  chunkSearching = true;
  chunkHasSearched = false;
  chunkHits = [];
  chunkOffset = offset;
  chunkActiveIndex = -1;
  renderChunkResults();
  updateChunkAvailability();

  try {
    const response = await api.searchChunks({
      query,
      documentId,
      matchCase: chunkMatchCase.checked,
      limit: CHUNK_PAGE_SIZE,
      offset,
    });
    if (!chunkGate.isCurrent(token) || selectedId !== documentId) {
      return;
    }
    chunkSearching = false;
    chunkHasSearched = true;
    chunkHits = response.results;
    chunkTotal = response.total;
    chunkOffset = response.offset;
    chunkActiveIndex = chunkHits.length > 0 ? 0 : -1;
    renderChunkResults();
    chunkHint.className = "search-hint is-info";
    chunkHint.textContent = `找到 ${
      response.total
    } 个匹配的 Chunk，共 ${Math.ceil(response.total / CHUNK_PAGE_SIZE)} 页。`;
  } catch (error) {
    if (!chunkGate.isCurrent(token)) {
      return;
    }
    chunkSearching = false;
    chunkHasSearched = false;
    chunkHits = [];
    chunkTotal = 0;
    chunkOffset = 0;
    chunkActiveIndex = -1;
    renderChunkResults();
    chunkHint.className = "search-hint is-error";
    chunkHint.textContent = friendlyError(error);
    showToast(friendlyError(error), "error");
  } finally {
    if (chunkGate.isCurrent(token)) {
      chunkSearchInFlight = false;
      chunkSearching = false;
      updateChunkAvailability();
    }
  }
}

function performChunkSearch(event: SubmitEvent): void {
  event.preventDefault();
  void runChunkSearch(0);
}

function goChunkPage(offset: number): void {
  void runChunkSearch(offset);
}

function renderResults(): void {
  resultsList.replaceChildren();
  resultCount.textContent = String(hits.length);
  hitPosition.textContent =
    activeHitIndex >= 0 && hits.length > 0
      ? `${activeHitIndex + 1} / ${hits.length}`
      : "—";
  hitPrev.disabled = hits.length === 0;
  hitNext.disabled = hits.length === 0;

  if (hits.length === 0) {
    const empty = document.createElement("div");
    empty.className = "list-empty";
    if (searching) {
      empty.textContent = "正在检索 PDF 原文…";
    } else if (hasSearched) {
      empty.textContent =
        "没有找到匹配证据。试试更短的片段，或关闭「区分大小写」。";
    } else {
      empty.textContent = "搜索后，命中的证据会按页码列在这里。";
    }
    resultsList.append(empty);
    return;
  }

  hits.forEach((hit, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `result-item${
      index === activeHitIndex ? " is-active" : ""
    }`;
    button.setAttribute(
      "aria-current",
      index === activeHitIndex ? "true" : "false"
    );

    const topLine = document.createElement("span");
    topLine.className = "result-topline";

    const pagePill = document.createElement("span");
    pagePill.className = "page-pill";
    pagePill.textContent = `第 ${hit.page_number} 页`;

    const charRange = document.createElement("span");
    charRange.className = "result-char";
    charRange.textContent = `字符 ${hit.char_start}–${
      hit.char_start + hit.char_count
    }`;

    topLine.append(pagePill, charRange);

    const text = document.createElement("p");
    text.className = "result-text";
    text.textContent = snippet(hit.matched_text);

    button.append(topLine, text);
    button.addEventListener("click", () => {
      void activateHit(index);
    });

    resultsList.append(button);
  });
}

async function activateHit(index: number): Promise<void> {
  if (index < 0 || index >= hits.length) {
    return;
  }

  activeHitIndex = index;
  highlightSelection = anchorSelection(hits, index);
  renderResults();

  const hit = hits[index];
  const requestToken = scrollGate.begin();
  await viewer.setPage(hit.page_number);

  // A newer click or a different active hit supersedes this request. The
  // viewer cancels stale renders itself; this guard prevents an older render
  // from scrolling the fresh page to a stale position.
  if (!scrollGate.isCurrent(requestToken) || activeHitIndex !== index) {
    return;
  }

  await centerActiveHighlight(requestToken);

  const activeItem = resultsList.querySelector<HTMLElement>(
    ".result-item.is-active"
  );
  if (activeItem) {
    revealWithin(resultsList, activeItem);
  }
}

async function centerActiveHighlight(requestToken: number): Promise<void> {
  await centerTargetsWhenReady({
    gate: scrollGate,
    requestToken,
    // `setPage()` resolves only after the target page finished rendering and
    // `onRender` painted the highlight layer. `waitForLayout()` lets the
    // browser apply those styles before measuring.
    wait: () => waitForLayout(pageShell),
    collectTargets: () =>
      Array.from(
        highlightLayer.querySelectorAll<HTMLElement>(".pdf-highlight--active")
      ),
    container: viewerStage,
  });
}

function moveHit(delta: number): void {
  if (hits.length === 0) {
    return;
  }
  const next = (activeHitIndex + delta + hits.length) % hits.length;
  void activateHit(next);
}

async function performSearch(event: SubmitEvent): Promise<void> {
  event.preventDefault();

  if (!selectedId || searchInFlight) {
    return;
  }

  const query = searchInput.value.trim();
  if (!query) {
    searchHint.className = "search-hint is-error";
    searchHint.textContent = "请输入要定位的证据文字。";
    searchInput.focus();
    return;
  }

  const documentId = selectedId;
  const token = ++searchToken;
  searchInFlight = true;
  searching = true;
  hasSearched = false;
  hits = [];
  activeHitIndex = -1;
  highlightSelection = emptySelection();
  searchButton.textContent = "搜索中…";
  updateSearchButton();
  searchHint.className = "search-hint is-info";
  searchHint.textContent = "正在检索 PDF 原文…";
  highlightLayer.replaceChildren();
  renderResults();

  try {
    const response = await api.searchAnchors(
      documentId,
      query,
      matchCase.checked
    );
    if (token !== searchToken || selectedId !== documentId) {
      return;
    }

    searching = false;
    hasSearched = true;
    hits = response.results;
    activeHitIndex = hits.length > 0 ? 0 : -1;
    renderResults();

    if (hits.length === 0) {
      searchHint.className = "search-hint";
      searchHint.textContent = "没有找到匹配证据。";
      highlightSelection = emptySelection();
      highlightLayer.replaceChildren();
    } else {
      searchHint.className = "search-hint is-info";
      searchHint.textContent = `找到 ${response.total} 处命中，已定位第 1 处。`;
      await activateHit(0);
    }
  } catch (error) {
    if (token !== searchToken) {
      return;
    }
    searching = false;
    hasSearched = false;
    hits = [];
    activeHitIndex = -1;
    highlightSelection = emptySelection();
    renderResults();
    searchHint.className = "search-hint is-error";
    searchHint.textContent = friendlyError(error);
    showToast(friendlyError(error), "error");
    if (error instanceof ApiError && error.status === 404) {
      await loadDocuments();
    }
  } finally {
    if (token === searchToken) {
      searchInFlight = false;
      searching = false;
      searchButton.textContent = "搜索";
      updateSearchButton();
    }
  }
}

/* ---------- Page and zoom controls ---------- */

prevPage.addEventListener("click", () => {
  void viewer.previousPage();
});

nextPage.addEventListener("click", () => {
  void viewer.nextPage();
});

pageInput.addEventListener("change", () => {
  const target = Number(pageInput.value);
  if (Number.isFinite(target)) {
    void viewer.setPage(target);
  }
});

pageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    const target = Number(pageInput.value);
    if (Number.isFinite(target)) {
      void viewer.setPage(target);
    }
    pageInput.blur();
  }
});

zoomIn.addEventListener("click", () => {
  void viewer.zoomIn();
});

zoomOut.addEventListener("click", () => {
  void viewer.zoomOut();
});

fitWidth.addEventListener("click", () => {
  void viewer.fitToWidth();
});

hitPrev.addEventListener("click", () => {
  moveHit(-1);
});

hitNext.addEventListener("click", () => {
  moveHit(1);
});

/* ---------- Form events ---------- */

searchForm.addEventListener("submit", (event) => {
  void performSearch(event);
});

searchInput.addEventListener("input", () => {
  updateSearchButton();
});

searchInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    searchForm.requestSubmit();
  }
});

/* ---------- Index and chunk events ---------- */

indexButton.addEventListener("click", () => {
  void performIndex();
});

chunkForm.addEventListener("submit", (event) => {
  performChunkSearch(event);
});

chunkPrev.addEventListener("click", () => {
  const page = computeChunkPage(chunkTotal, chunkOffset, chunkHits.length);
  goChunkPage(page.previousOffset);
});

chunkNext.addEventListener("click", () => {
  const page = computeChunkPage(chunkTotal, chunkOffset, chunkHits.length);
  goChunkPage(page.nextOffset);
});

chunkInput.addEventListener("input", () => {
  updateChunkAvailability();
});

/* ---------- Bootstrap ---------- */

window.addEventListener("beforeunload", () => {
  viewer.destroy();
});

documentCount.textContent = "0";
renderResults();
renderChunkResults();
updateSearchAvailability();
void loadDocuments();
