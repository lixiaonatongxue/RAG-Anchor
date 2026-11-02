import { ApiError, RagAnchorApi } from "./api/client";
import type {
  AnchorHit,
  AnswerMode,
  CitationSupportKind,
  DocumentIndexResponse,
  DocumentSummary,
  EvidenceCitation,
  QuestionAnswerResponse,
  RetrievalHit,
  RetrievalMode,
  SemanticIndexResponse,
} from "./api/types";
import {
  clearHighlightFor,
  emptyHighlightState,
  highlightFor,
} from "./highlight/ownership";
import type { HighlightSource, HighlightState } from "./highlight/ownership";
import { paintHighlights } from "./highlight/renderer";
import {
  anchorSelection,
  evidenceSelection,
  selectPageHighlights,
} from "./highlight/selection";
import type { HighlightSelection } from "./highlight/selection";
import { PdfViewer } from "./pdf/viewer";
import type { ViewerState } from "./pdf/viewer";
import {
  citationBelongsToDocument,
  citationLabel,
  citationSupportLabel,
  citationSupportSpan,
  toCitationEvidenceSpan,
} from "./qa/citation";
import {
  generationFailureMessage,
  semanticFailureMessage,
  semanticIndexFailureMessage,
} from "./qa/mode";
import { insufficientEvidenceMessage } from "./qa/refusal";
import { isCurrentQaResponse } from "./qa/request";
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

const libraryPane = byId<HTMLElement>("library-pane");
const libraryToggle = byId<HTMLButtonElement>("library-toggle");
const libraryClose = byId<HTMLButtonElement>("library-close");
const inspectorPane = byId<HTMLElement>("inspector-pane");
const inspectorToggle = byId<HTMLButtonElement>("inspector-toggle");
const inspectorClose = byId<HTMLButtonElement>("inspector-close");
const drawerBackdrop = byId<HTMLDivElement>("drawer-backdrop");
const narrowLayout = window.matchMedia("(max-width: 1024px)");

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

const qaForm = byId<HTMLFormElement>("qa-form");
const qaRetrievalMode = byId<HTMLSelectElement>("qa-retrieval-mode");
const qaMode = byId<HTMLSelectElement>("qa-mode");
const qaQuestion = byId<HTMLTextAreaElement>("qa-question");
const qaQuery = byId<HTMLInputElement>("qa-query");
const qaSubmit = byId<HTMLButtonElement>("qa-submit");
const qaHint = byId<HTMLParagraphElement>("qa-hint");
const qaEmpty = byId<HTMLDivElement>("qa-empty");
const qaSummary = byId<HTMLElement>("qa-summary");
const qaSummaryTitle = byId<HTMLHeadingElement>("qa-summary-title");
const qaSummaryCount = byId<HTMLSpanElement>("qa-summary-count");
const qaSummaryNote = byId<HTMLParagraphElement>("qa-summary-note");
const qaAnswerDetails = byId<HTMLDetailsElement>("qa-answer-details");
const qaAnswerDetailsText = byId<HTMLParagraphElement>("qa-answer-details-text");
const qaAnswer = byId<HTMLDivElement>("qa-answer");
const qaAnswerText = byId<HTMLParagraphElement>("qa-answer-text");
const qaCitations = byId<HTMLDivElement>("qa-citations");
const semanticIndexButton = byId<HTMLButtonElement>("semantic-index-button");
const semanticIndexDetail = byId<HTMLSpanElement>("semantic-index-detail");

const searchForm = byId<HTMLFormElement>("search-form");
const searchInput = byId<HTMLTextAreaElement>("search-input");
const matchCase = byId<HTMLInputElement>("match-case");
const searchButton = byId<HTMLButtonElement>("search-button");
const searchHint = byId<HTMLParagraphElement>("search-hint");
const searchResultsPanel = byId<HTMLElement>("search-results");

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
 * The single highlight input for the overlay. Anchor search, chunk evidence
 * and QA citation location all publish into this state, so there is exactly
 * one coordinate conversion path (`highlight/renderer.ts`). The source tag
 * lets QA cleanup clear its own citation without removing an unrelated anchor
 * or chunk highlight.
 */
let highlightState: HighlightState = emptyHighlightState();
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

/* ---------- Evidence-only question answering state ---------- */

let qaCitationItems: EvidenceCitation[] = [];
let qaActiveIndex = -1;
let qaInFlight = false;
const qaGate = createRequestGate();

let semanticIndexedDocumentId: string | null = null;
let semanticIndexStats: SemanticIndexResponse | null = null;
let semanticIndexInFlight = false;
const semanticIndexGate = createRequestGate();

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

const TOAST_LIMIT = 3;
const toastTimers = new WeakMap<HTMLElement, number>();

function dismissToast(toast: HTMLElement): void {
  const timer = toastTimers.get(toast);
  if (timer !== undefined) {
    window.clearTimeout(timer);
    toastTimers.delete(toast);
  }
  if (toast.classList.contains("is-leaving")) {
    return;
  }
  toast.classList.add("is-leaving");
  window.setTimeout(() => toast.remove(), 180);
}

function armToastTimer(toast: HTMLElement, duration: number): void {
  const previous = toastTimers.get(toast);
  if (previous !== undefined) {
    window.clearTimeout(previous);
  }
  const timer = window.setTimeout(() => {
    toastTimers.delete(toast);
    dismissToast(toast);
  }, duration);
  toastTimers.set(toast, timer);
}

function showToast(
  message: string,
  kind: "info" | "success" | "error" = "info"
): void {
  const toastKey = `${kind}:${message}`;
  const existing = Array.from(toastRegion.children).find(
    (node) =>
      node instanceof HTMLElement &&
      node.dataset.toastKey === toastKey &&
      !node.classList.contains("is-leaving")
  ) as HTMLElement | undefined;
  if (existing) {
    if (kind !== "error") {
      armToastTimer(existing, kind === "success" ? 3000 : 3200);
    }
    return;
  }

  while (toastRegion.childElementCount >= TOAST_LIMIT) {
    const children = Array.from(toastRegion.children) as HTMLElement[];
    // Keep failures visible; only evict a remaining error when every toast is
    // an error, which still bounds the stack.
    const victim =
      children.find((node) => node.dataset.toastKind !== "error") ?? children[0];
    const timer = toastTimers.get(victim);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      toastTimers.delete(victim);
    }
    victim.remove();
  }

  const toast = document.createElement("div");
  toast.className = `toast toast--${kind}`;
  toast.dataset.toastKey = toastKey;
  toast.dataset.toastKind = kind;
  toast.setAttribute("role", kind === "error" ? "alert" : "status");
  toast.setAttribute("aria-atomic", "true");

  const text = document.createElement("span");
  text.className = "toast-text";
  text.textContent = message;
  toast.append(text);

  if (kind === "error") {
    const close = document.createElement("button");
    close.type = "button";
    close.className = "toast-close";
    close.setAttribute("aria-label", "关闭通知");
    close.textContent = "×";
    close.addEventListener("click", () => dismissToast(toast));
    toast.append(close);
  } else {
    armToastTimer(toast, kind === "success" ? 3000 : 3200);
  }

  toastRegion.append(toast);
}

function setConnection(state: "idle" | "ok" | "error", label: string): void {
  connectionStatus.dataset.state = state;
  connectionLabel.textContent = label;
}

/* ---------- Drawer panes ---------- */

function setDrawerOpen(
  pane: HTMLElement,
  toggle: HTMLButtonElement,
  open: boolean
): void {
  pane.classList.toggle("is-open", open);
  toggle.setAttribute("aria-expanded", String(open));
}

function closeDrawers(): void {
  setDrawerOpen(libraryPane, libraryToggle, false);
  setDrawerOpen(inspectorPane, inspectorToggle, false);
  drawerBackdrop.hidden = true;
}

function toggleDrawer(
  pane: HTMLElement,
  toggle: HTMLButtonElement,
  otherPane: HTMLElement,
  otherToggle: HTMLButtonElement
): void {
  const open = !pane.classList.contains("is-open");
  setDrawerOpen(otherPane, otherToggle, false);
  setDrawerOpen(pane, toggle, open);
  drawerBackdrop.hidden = !open;
}

function toggleLibraryDrawer(): void {
  toggleDrawer(libraryPane, libraryToggle, inspectorPane, inspectorToggle);
}

function toggleInspectorDrawer(): void {
  toggleDrawer(inspectorPane, inspectorToggle, libraryPane, libraryToggle);
}

libraryToggle.addEventListener("click", toggleLibraryDrawer);
libraryClose.addEventListener("click", closeDrawers);
inspectorToggle.addEventListener("click", toggleInspectorDrawer);
inspectorClose.addEventListener("click", closeDrawers);
drawerBackdrop.addEventListener("click", closeDrawers);
narrowLayout.addEventListener("change", (event) => {
  if (!event.matches) {
    closeDrawers();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeDrawers();
  }
});

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
  const page = selectPageHighlights(highlightState.selection, pageNumber);
  paintHighlights(highlightLayer, page.hits, viewport, page.activeLocalIndex);
}

function setHighlight(
  source: HighlightSource,
  selection: HighlightSelection
): void {
  highlightState = highlightFor(source, selection);
}

/** Clear the overlay, or only one source's highlight when `source` is given. */
function clearHighlight(source?: HighlightSource): void {
  const next = clearHighlightFor(highlightState, source);
  if (next === highlightState) {
    return;
  }
  highlightState = next;
  highlightLayer.replaceChildren();
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
    empty.textContent = "暂无文献";
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

    body.append(name, meta);

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
  if (narrowLayout.matches) {
    closeDrawers();
  }
  resetSearchResults();
  resetChunkState();
  resetIndexState();
  resetQaState();
  updateSearchAvailability();

  viewerTitle.textContent = document.filename;
  viewerTitle.title = document.filename;
  viewerSubtitle.textContent = `${document.page_count} 页 · ${formatBytes(
    document.size_bytes
  )}`;
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
    viewerTitle.title = detail.filename;
    viewerSubtitle.textContent = `${detail.page_count} 页 · ${formatBytes(
      detail.size_bytes
    )}`;

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
  resetQaState();
  updateSearchAvailability();
  renderDocuments();

  viewerTitle.textContent = "未选择文档";
  viewerTitle.title = "未选择文档";
  viewerSubtitle.textContent = "";
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
    searchHint.textContent = "请先选择文献。";
  } else if (!hasSearched && !searchInFlight) {
    searchHint.className = "search-hint";
    searchHint.textContent = "";
  }

  updateSearchButton();
  updateIndexAvailability();
  updateChunkAvailability();
  updateQaAvailability();
}

function resetSearchResults(): void {
  hits = [];
  activeHitIndex = -1;
  clearHighlight();
  hasSearched = false;
  searching = false;
  searchToken += 1;
  searchInFlight = false;
  searchButton.textContent = "搜索";
  renderResults();
  updateSearchAvailability();
}

/* ---------- Index and chunk retrieval ---------- */

function resetIndexState(): void {
  indexedDocumentId = null;
  indexStats = null;
  indexInFlight = false;
  indexStatus.textContent = "未索引";
  resetSemanticIndexState();
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
  chunkHint.textContent = selectedId ? "请先建立索引。" : "请先选择文献。";
  chunkPrev.disabled = true;
  chunkNext.disabled = true;
  chunkSearchButton.disabled = true;
  renderChunkResults();
}

function currentRetrievalMode(): RetrievalMode {
  return qaRetrievalMode.value === "semantic" ? "semantic" : "literal";
}

function currentQaMode(): AnswerMode {
  return qaMode.value === "grounded_generation"
    ? "grounded_generation"
    : "evidence_only";
}

function semanticIndexReady(): boolean {
  return (
    Boolean(selectedId) &&
    indexedDocumentId === selectedId &&
    semanticIndexedDocumentId === selectedId &&
    semanticIndexStats !== null
  );
}

function qaIdleHint(): string {
  if (!selectedId) {
    return "请先选择文献。";
  }
  if (indexedDocumentId !== selectedId) {
    return "请先建立索引。";
  }
  if (currentRetrievalMode() === "semantic" && !semanticIndexReady()) {
    return "请先建立语义索引。";
  }
  return "";
}

function uniformCitationSupport(
  citations: EvidenceCitation[]
): CitationSupportKind | null {
  if (citations.length === 0) {
    return null;
  }
  const first = citationSupportSpan(citations[0]).kind;
  return citations.every(
    (citation) => citationSupportSpan(citation).kind === first
  )
    ? first
    : null;
}

function resetQaOutput(): void {
  qaSummary.hidden = true;
  qaSummaryTitle.textContent = "检索结果";
  qaSummaryCount.textContent = "";
  qaSummaryNote.textContent = "";
  qaAnswerDetails.hidden = true;
  qaAnswerDetails.open = false;
  qaAnswerDetailsText.textContent = "";
  qaAnswer.hidden = true;
  qaAnswerText.textContent = "";
  qaCitations.replaceChildren();
}

function renderQaSummary(
  response: QuestionAnswerResponse,
  mode: AnswerMode
): void {
  const uniform = uniformCitationSupport(response.citations);
  const supportSuffix = uniform
    ? ` · 均为${citationSupportLabel(uniform)}`
    : "";
  const matchCount =
    response.retrieval_mode === "semantic"
      ? `${response.total_matches} 条候选`
      : `${response.total_matches} 处匹配`;

  qaSummaryTitle.textContent =
    mode === "grounded_generation" ? "AI 生成回答" : "检索结果";
  qaSummaryCount.textContent =
    mode === "grounded_generation"
      ? `${matchCount} · ${response.citations.length} 条引用`
      : `${matchCount} · ${response.citations.length} 条证据`;
  qaSummaryNote.textContent =
    mode === "grounded_generation"
      ? `引用编号由服务端校验，结论仍需人工核对原文${supportSuffix}`
      : `仅展示原文证据，未调用语言模型${supportSuffix}`;
  qaSummary.hidden = false;

  if (mode === "evidence_only" && response.answer.trim()) {
    qaAnswerDetailsText.textContent = response.answer;
    qaAnswerDetails.hidden = false;
  }
}

function resetQaState(): void {
  qaGate.begin();
  locateGate.begin();
  qaCitationItems = [];
  qaActiveIndex = -1;
  clearHighlight("qa");
  qaInFlight = false;
  resetQaOutput();
  qaEmpty.hidden = false;
  qaHint.className = "search-hint";
  qaHint.textContent = qaIdleHint();
  updateQaAvailability();
}

function updateQaAvailability(): void {
  const indexed = Boolean(selectedId) && indexedDocumentId === selectedId;
  const semanticReady =
    currentRetrievalMode() === "literal" || semanticIndexReady();
  qaRetrievalMode.disabled = qaInFlight || semanticIndexInFlight;
  qaMode.disabled = qaInFlight || semanticIndexInFlight;
  qaQuery.disabled = qaInFlight || currentRetrievalMode() === "semantic";
  qaSubmit.disabled =
    !indexed ||
    !semanticReady ||
    indexInFlight ||
    semanticIndexInFlight ||
    qaInFlight ||
    qaQuestion.value.trim().length === 0;
  if (qaInFlight) {
    qaSubmit.textContent =
      currentQaMode() === "grounded_generation" ? "生成中…" : "检索中…";
  } else {
    qaSubmit.textContent =
      currentQaMode() === "grounded_generation" ? "生成回答" : "提问";
  }
}

function updateSemanticAvailability(): void {
  const document = documents.find((item) => item.document_id === selectedId);
  if (!document) {
    semanticIndexButton.disabled = true;
    semanticIndexButton.textContent = "建立语义索引";
    semanticIndexDetail.textContent = "请先选择文献";
    semanticIndexDetail.title = "请先选择文献";
    return;
  }
  if (semanticIndexInFlight) {
    semanticIndexButton.disabled = true;
    semanticIndexButton.textContent = "语义索引建立中…";
    semanticIndexDetail.textContent = "正在建立语义索引…";
    semanticIndexDetail.title = "";
    return;
  }
  if (indexInFlight) {
    semanticIndexButton.disabled = true;
    semanticIndexButton.textContent = "建立语义索引";
    semanticIndexDetail.textContent = "等待文本索引完成";
    semanticIndexDetail.title = "";
    return;
  }
  if (qaInFlight) {
    const built =
      semanticIndexedDocumentId === document.document_id && semanticIndexStats;
    semanticIndexButton.disabled = true;
    semanticIndexButton.textContent = built ? "重建语义索引" : "建立语义索引";
    semanticIndexDetail.textContent = "等待问答完成";
    semanticIndexDetail.title = "";
    return;
  }
  if (indexedDocumentId !== document.document_id) {
    semanticIndexButton.disabled = true;
    semanticIndexButton.textContent = "建立语义索引";
    semanticIndexDetail.textContent = "需先建立文本索引";
    semanticIndexDetail.title = "";
    return;
  }
  if (
    semanticIndexedDocumentId === document.document_id &&
    semanticIndexStats
  ) {
    semanticIndexButton.disabled = false;
    semanticIndexButton.textContent = "重建语义索引";
    semanticIndexDetail.textContent = `已建立 · ${semanticIndexStats.candidate_count} 候选 · ${semanticIndexStats.dimension} 维`;
    semanticIndexDetail.title = `${semanticIndexStats.model_name}/${semanticIndexStats.model_version} · ${semanticIndexStats.dimension} 维 · ${semanticIndexStats.candidate_count} 个候选`;
    return;
  }
  semanticIndexButton.disabled = false;
  semanticIndexButton.textContent = "建立语义索引";
  semanticIndexDetail.textContent = "";
  semanticIndexDetail.title = "将调用 Embedding 服务生成语义索引";
}

function resetSemanticIndexState(): void {
  semanticIndexGate.begin();
  semanticIndexedDocumentId = null;
  semanticIndexStats = null;
  semanticIndexInFlight = false;
  updateSemanticAvailability();
}

function updateIndexAvailability(): void {
  const document = documents.find((item) => item.document_id === selectedId);
  if (!document) {
    indexButton.disabled = true;
    indexButton.textContent = "建立索引";
    indexDetail.textContent = "请先选择文献";
    indexDetail.title = "请先选择文献";
    return;
  }
  if (indexInFlight) {
    indexButton.disabled = true;
    indexButton.textContent = "索引中…";
    indexDetail.textContent = "正在建立索引…";
    indexDetail.title = "";
    return;
  }
  if (indexedDocumentId === document.document_id && indexStats) {
    indexButton.disabled = false;
    indexButton.textContent = "重新索引";
    indexDetail.textContent = `${indexStats.page_count} 页 · ${indexStats.chunk_count} Chunk`;
    indexDetail.title = `已索引 ${indexStats.page_count} 页 · ${indexStats.chunk_count} 个 Chunk`;
    return;
  }
  indexButton.disabled = false;
  indexButton.textContent = "建立索引";
  indexDetail.textContent = "";
  indexDetail.title = "";
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
  // Re-indexing can replace chunk identifiers, so old answers and any
  // in-flight question must not survive into the new index. The backend also
  // invalidates the derived semantic index, which must be rebuilt.
  resetQaState();
  resetSemanticIndexState();

  try {
    const stats = await api.indexDocument(documentId);
    if (selectedId !== documentId) {
      return;
    }
    indexedDocumentId = documentId;
    indexStats = stats;
    indexStatus.textContent = "已索引";
    qaHint.className = "search-hint";
    qaHint.textContent = qaIdleHint();
    showToast(
      `已索引 ${stats.page_count} 页 · ${stats.chunk_count} Chunk`,
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
      updateSemanticAvailability();
      updateChunkAvailability();
      updateQaAvailability();
    }
  }
}

async function performSemanticIndex(): Promise<void> {
  const documentId = selectedId;
  if (
    !documentId ||
    semanticIndexInFlight ||
    qaInFlight ||
    indexInFlight ||
    indexedDocumentId !== documentId
  ) {
    return;
  }

  const token = semanticIndexGate.begin();
  // The rebuild replaces the vector index and every candidate citation it
  // produced. Invalidate an in-flight answer and clear the displayed one before
  // the request starts, so a late response cannot restore the old answer.
  resetQaState();
  semanticIndexInFlight = true;
  updateSemanticAvailability();
  updateQaAvailability();
  qaHint.className = "search-hint is-info";
  qaHint.textContent = "正在建立语义索引…";

  try {
    const stats = await api.buildSemanticIndex(documentId);
    if (
      !semanticIndexGate.isCurrent(token) ||
      selectedId !== documentId ||
      indexedDocumentId !== documentId
    ) {
      return;
    }
    semanticIndexedDocumentId = documentId;
    semanticIndexStats = stats;
    showToast(
      `已建立语义索引 · ${stats.candidate_count} 候选`,
      "success"
    );
    qaHint.className = "search-hint";
    qaHint.textContent = qaIdleHint();
  } catch (error) {
    if (!semanticIndexGate.isCurrent(token) || selectedId !== documentId) {
      return;
    }
    const status = error instanceof ApiError ? error.status : 0;
    const message =
      semanticIndexFailureMessage(status) ?? friendlyError(error);
    qaHint.className = "search-hint is-error";
    qaHint.textContent = message;
    showToast(message, "error");
  } finally {
    if (semanticIndexGate.isCurrent(token) && selectedId === documentId) {
      semanticIndexInFlight = false;
      updateSemanticAvailability();
      updateQaAvailability();
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

    setHighlight("chunk", evidenceSelection(located));
    await viewer.setPage(located.page_number);

    const scrollToken = scrollGate.begin();
    await centerActiveHighlight(scrollToken);
    if (narrowLayout.matches) {
      closeDrawers();
    }
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
    chunkHint.textContent = `找到 ${response.total} 个 Chunk。`;
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

function renderQaCitations(): void {
  qaCitations.replaceChildren();
  const uniform = uniformCitationSupport(qaCitationItems);

  qaCitationItems.forEach((citation, index) => {
    const support = citationSupportSpan(citation);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `result-item qa-citation${
      index === qaActiveIndex ? " is-active" : ""
    }`;
    button.setAttribute(
      "aria-current",
      index === qaActiveIndex ? "true" : "false"
    );
    button.title = citationLabel(citation);
    button.setAttribute(
      "aria-label",
      `${citationLabel(citation)}：${snippet(support.text, 80)}`
    );

    const meta = document.createElement("span");
    meta.className = "citation-meta";

    const citationId = document.createElement("span");
    citationId.className = "citation-id";
    citationId.textContent = citation.citation_id;

    const separator = document.createElement("span");
    separator.className = "citation-separator";
    separator.textContent = "·";
    separator.setAttribute("aria-hidden", "true");

    const page = document.createElement("span");
    page.className = "citation-page";
    page.textContent = `第 ${citation.page_number} 页`;

    meta.append(citationId, separator, page);

    if (uniform === null) {
      const supportPill = document.createElement("span");
      supportPill.className = `support-pill support-pill--${support.kind}`;
      supportPill.textContent = citationSupportLabel(support.kind);
      meta.append(supportPill);
    }
    if (
      support.kind === "semantic_candidate" &&
      typeof citation.score === "number"
    ) {
      const score = document.createElement("span");
      score.className = "citation-score";
      score.textContent = `相似度 ${citation.score.toFixed(3)}`;
      meta.append(score);
    }

    const hit = document.createElement("p");
    hit.className = "citation-hit";
    hit.textContent = snippet(citation.hit_text, 180);

    const context = document.createElement("p");
    context.className = "citation-context";
    context.textContent = snippet(citation.chunk_text, 220);
    if (citation.chunk_text.length > 220) {
      context.title = citation.chunk_text;
    }

    button.append(meta, hit, context);
    button.addEventListener("click", () => {
      void locateQaCitation(index);
    });
    qaCitations.append(button);
  });
}

async function performQuestion(event: SubmitEvent): Promise<void> {
  event.preventDefault();

  const documentId = selectedId;
  if (!documentId || qaInFlight || semanticIndexInFlight) {
    return;
  }
  if (indexedDocumentId !== documentId) {
    qaHint.className = "search-hint is-error";
    qaHint.textContent = "请先建立索引。";
    return;
  }

  const question = qaQuestion.value.trim();
  if (!question) {
    qaHint.className = "search-hint is-error";
    qaHint.textContent = "请输入问题。";
    qaQuestion.focus();
    return;
  }

  const retrievalQuery = qaQuery.value.trim();
  const mode = currentQaMode();
  const retrievalMode = currentRetrievalMode();
  if (retrievalMode === "semantic" && !semanticIndexReady()) {
    qaHint.className = "search-hint is-error";
    qaHint.textContent = "请先建立语义索引。";
    return;
  }
  const token = qaGate.begin();
  locateGate.begin();
  const requestContext = { documentId, answerMode: mode, retrievalMode };
  const responseIsCurrent = (): boolean =>
    isCurrentQaResponse(qaGate, token, requestContext, {
      documentId: selectedId,
      answerMode: currentQaMode(),
      retrievalMode: currentRetrievalMode(),
    });
  clearHighlight("qa");
  qaInFlight = true;
  qaCitationItems = [];
  qaActiveIndex = -1;
  resetQaOutput();
  qaEmpty.hidden = true;
  qaHint.className = "search-hint is-info";
  if (mode === "grounded_generation") {
    qaHint.textContent = "正在生成回答…";
  } else if (retrievalMode === "semantic") {
    qaHint.textContent = "正在检索语义候选…";
  } else {
    qaHint.textContent = "正在检索证据…";
  }
  updateQaAvailability();

  try {
    const response = await api.askQuestion(documentId, {
      question,
      retrievalQuery:
        retrievalMode === "literal" ? retrievalQuery || undefined : undefined,
      retrievalMode,
      answerMode: mode,
    });
    if (!responseIsCurrent()) {
      return;
    }
    if (
      response.answer_mode !== mode ||
      response.retrieval_mode !== retrievalMode
    ) {
      throw new Error("后端返回了与请求不一致的问答或检索模式。");
    }

    qaInFlight = false;
    qaCitationItems = response.citations;
    qaActiveIndex = -1;
    resetQaOutput();
    renderQaSummary(response, mode);
    if (mode === "grounded_generation" && response.answer.trim()) {
      qaAnswerText.textContent = response.answer;
      qaAnswer.hidden = false;
    }
    renderQaCitations();

    if (response.status === "insufficient_evidence") {
      qaHint.className = "search-hint";
      qaHint.textContent = insufficientEvidenceMessage(
        response,
        retrievalMode,
        mode
      );
    } else {
      qaHint.className = "search-hint";
      qaHint.textContent = "";
    }
  } catch (error) {
    if (!responseIsCurrent()) {
      return;
    }
    qaInFlight = false;
    qaCitationItems = [];
    qaActiveIndex = -1;
    resetQaOutput();
    qaEmpty.hidden = false;
    const status = error instanceof ApiError ? error.status : 0;
    const fallback = friendlyError(error);
    const message =
      semanticFailureMessage(status, retrievalMode) ??
      (mode === "grounded_generation"
        ? generationFailureMessage(status)
        : null) ??
      fallback;
    qaHint.className = "search-hint is-error";
    qaHint.textContent = message;
    showToast(message, "error");
  } finally {
    if (responseIsCurrent()) {
      qaInFlight = false;
      updateQaAvailability();
    }
  }
}

async function locateQaCitation(index: number): Promise<void> {
  const citation = qaCitationItems[index];
  const documentId = selectedId;
  if (!citation || !documentId) {
    return;
  }
  if (!citationBelongsToDocument(citation, documentId)) {
    qaHint.className = "search-hint is-error";
    qaHint.textContent = "引用不属于当前文档，已拒绝定位。";
    return;
  }

  qaActiveIndex = index;
  renderQaCitations();

  const requestToken = locateGate.begin();
  try {
    const response = await api.locateEvidence(documentId, [
      toCitationEvidenceSpan(citation),
    ]);
    if (!locateGate.isCurrent(requestToken) || selectedId !== documentId) {
      return;
    }
    const located = response.results[0];
    if (!located) {
      clearHighlight("qa");
      qaActiveIndex = -1;
      renderQaCitations();
      qaHint.className = "search-hint is-error";
      qaHint.textContent = "服务端没有返回定位结果。";
      return;
    }

    setHighlight("qa", evidenceSelection(located));
    await viewer.setPage(located.page_number);

    const scrollToken = scrollGate.begin();
    await centerActiveHighlight(scrollToken);
    if (!locateGate.isCurrent(requestToken) || selectedId !== documentId) {
      return;
    }

    qaHint.className = "search-hint is-info";
    qaHint.textContent = `已定位 ${citation.citation_id}：第 ${located.page_number} 页。`;
    const activeItem = qaCitations.querySelector<HTMLElement>(
      ".qa-citation.is-active"
    );
    if (activeItem) {
      revealWithin(qaCitations, activeItem);
    }
    if (narrowLayout.matches) {
      closeDrawers();
    }
  } catch (error) {
    if (!locateGate.isCurrent(requestToken) || selectedId !== documentId) {
      return;
    }
    // A failed location must not leave this citation's highlight behind.
    // Other highlight sources are untouched. No fallback text search is
    // attempted.
    qaActiveIndex = -1;
    renderQaCitations();
    clearHighlight("qa");
    const status = error instanceof ApiError ? error.status : 0;
    const message = evidenceErrorMessage(status, friendlyError(error));
    qaHint.className = "search-hint is-error";
    qaHint.textContent = message;
    showToast(message, "error");
  }
}

function renderResults(): void {
  resultsList.replaceChildren();
  searchResultsPanel.classList.toggle("is-visible", searching || hasSearched);
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
      empty.textContent = "正在检索…";
    } else if (hasSearched) {
      empty.textContent = "没有找到匹配证据。";
    } else {
      empty.textContent = "暂无结果";
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
  setHighlight("anchor", anchorSelection(hits, index));
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
    searchHint.textContent = "请输入检索文字。";
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
  clearHighlight();
  searchButton.textContent = "搜索中…";
  updateSearchButton();
  searchHint.className = "search-hint is-info";
  searchHint.textContent = "正在检索…";
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
      clearHighlight();
    } else {
      searchHint.className = "search-hint is-info";
      searchHint.textContent = `找到 ${response.total} 处命中。`;
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
    clearHighlight();
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

semanticIndexButton.addEventListener("click", () => {
  void performSemanticIndex();
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

/* ---------- Question answering events ---------- */

qaForm.addEventListener("submit", (event) => {
  void performQuestion(event);
});

qaQuestion.addEventListener("input", () => {
  updateQaAvailability();
});

qaMode.addEventListener("change", () => {
  // A different mode must not inherit citations or answers from the previous
  // one, and any in-flight request belongs to the old mode.
  resetQaState();
});

qaRetrievalMode.addEventListener("change", () => {
  resetQaState();
});

/* ---------- Bootstrap ---------- */

window.addEventListener("beforeunload", () => {
  viewer.destroy();
});

resetQaState();
updateSemanticAvailability();
documentCount.textContent = "0";
renderResults();
renderChunkResults();
updateSearchAvailability();
void loadDocuments();
