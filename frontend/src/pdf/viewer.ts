import * as pdfjsLib from "pdfjs-dist";
import type {
  PDFDocumentLoadingTask,
  PDFDocumentProxy,
  PageViewport,
  RenderTask,
} from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;

const MIN_SCALE = 0.3;
const MAX_SCALE = 4;
const ZOOM_STEP = 1.15;
const STAGE_PADDING = 64;

export interface ViewerState {
  pageNumber: number;
  pageCount: number;
  scale: number;
  rendering: boolean;
  fitWidth: boolean;
  loaded: boolean;
}

export interface ViewerCallbacks {
  onStateChange: (state: ViewerState) => void;
  onRender: (pageNumber: number, viewport: PageViewport) => void;
  onError: (error: unknown) => void;
}

export interface ViewerElements {
  stage: HTMLElement;
  shell: HTMLElement;
  canvas: HTMLCanvasElement;
  overlay: HTMLElement;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function isRenderingCancelled(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    "name" in error &&
    (error as { name?: unknown }).name === "RenderingCancelledException"
  );
}

/**
 * Single-page PDF.js viewer with fit-width, zoom and stale-render protection.
 *
 * The PDF is loaded from an ArrayBuffer because M0-05B deliberately does not
 * depend on HTTP Range behaviour. Highlight rendering is delegated to the
 * caller through `onRender`, which receives the exact viewport used to paint
 * the canvas.
 */
export class PdfViewer {
  private readonly stage: HTMLElement;
  private readonly shell: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly overlay: HTMLElement;
  private readonly callbacks: ViewerCallbacks;
  private readonly context: CanvasRenderingContext2D | null;

  private loadingTask: PDFDocumentLoadingTask | null = null;
  private document: PDFDocumentProxy | null = null;
  private renderTask: RenderTask | null = null;
  private currentViewport: PageViewport | null = null;
  private renderToken = 0;
  private pageNumber = 1;
  private pageCount = 0;
  private scale = 1;
  private fitWidth = true;

  private readonly resizeObserver: ResizeObserver | null;
  private resizeTimer: number | null = null;

  constructor(elements: ViewerElements, callbacks: ViewerCallbacks) {
    this.stage = elements.stage;
    this.shell = elements.shell;
    this.canvas = elements.canvas;
    this.overlay = elements.overlay;
    this.callbacks = callbacks;
    this.context = elements.canvas.getContext("2d");

    this.resizeObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(() => {
            this.handleResize();
          });
    this.resizeObserver?.observe(this.stage);
  }

  get isLoaded(): boolean {
    return this.document !== null;
  }

  /** Load a PDF from memory after tearing down the previous document. */
  async open(data: ArrayBuffer): Promise<void> {
    await this.close();

    const token = this.renderToken;
    this.fitWidth = true;
    this.pageNumber = 1;

    const loadingTask = pdfjsLib.getDocument({
      data: new Uint8Array(data),
      isEvalSupported: false,
    });
    this.loadingTask = loadingTask;

    try {
      const document = await loadingTask.promise;
      if (token !== this.renderToken) {
        await loadingTask.destroy();
        return;
      }

      this.document = document;
      this.pageCount = document.numPages;
      this.pageNumber = 1;
      this.emitState(true);
      await this.renderCurrentPage();
    } catch (error) {
      if (this.loadingTask === loadingTask) {
        this.loadingTask = null;
      }
      try {
        await loadingTask.destroy();
      } catch {
        // Never mask the original loading failure with a cleanup failure.
      }
      throw error;
    }
  }

  /** Destroy the current document and clear the page/highlight layers. */
  async close(): Promise<void> {
    this.renderToken += 1;
    this.cancelRenderTask();

    const task = this.loadingTask;
    this.loadingTask = null;
    this.document = null;
    this.currentViewport = null;
    this.pageCount = 0;
    this.pageNumber = 1;
    this.scale = 1;
    this.fitWidth = true;

    this.overlay.replaceChildren();
    this.overlay.style.width = "0px";
    this.overlay.style.height = "0px";

    if (this.context) {
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.canvas.style.width = "0px";
    this.canvas.style.height = "0px";
    this.shell.style.width = "0px";
    this.shell.style.height = "0px";

    if (task) {
      try {
        await task.destroy();
      } catch {
        // Destroying an already-broken worker must not surface to the UI.
      }
    }

    this.emitState(false);
  }

  async setPage(pageNumber: number): Promise<void> {
    if (!this.document) {
      return;
    }

    const target = clamp(Math.trunc(pageNumber), 1, this.pageCount);
    if (target === this.pageNumber && this.currentViewport) {
      this.callbacks.onRender(this.pageNumber, this.currentViewport);
      return;
    }

    this.pageNumber = target;
    await this.renderCurrentPage();
  }

  async nextPage(): Promise<void> {
    if (this.pageNumber < this.pageCount) {
      await this.setPage(this.pageNumber + 1);
    }
  }

  async previousPage(): Promise<void> {
    if (this.pageNumber > 1) {
      await this.setPage(this.pageNumber - 1);
    }
  }

  async zoomIn(): Promise<void> {
    await this.zoomBy(ZOOM_STEP);
  }

  async zoomOut(): Promise<void> {
    await this.zoomBy(1 / ZOOM_STEP);
  }

  async fitToWidth(): Promise<void> {
    if (!this.document) {
      return;
    }
    this.fitWidth = true;
    await this.renderCurrentPage();
  }

  destroy(): void {
    this.resizeObserver?.disconnect();
    if (this.resizeTimer !== null) {
      window.clearTimeout(this.resizeTimer);
      this.resizeTimer = null;
    }
    void this.close();
  }

  private async zoomBy(factor: number): Promise<void> {
    if (!this.document) {
      return;
    }
    const current = this.currentViewport?.scale ?? this.scale;
    this.scale = clamp(current * factor, MIN_SCALE, MAX_SCALE);
    this.fitWidth = false;
    await this.renderCurrentPage();
  }

  private handleResize(): void {
    if (!this.document || !this.fitWidth) {
      return;
    }
    if (this.resizeTimer !== null) {
      window.clearTimeout(this.resizeTimer);
    }
    this.resizeTimer = window.setTimeout(() => {
      this.resizeTimer = null;
      void this.renderCurrentPage();
    }, 150);
  }

  private cancelRenderTask(): void {
    if (!this.renderTask) {
      return;
    }
    try {
      this.renderTask.cancel();
    } catch {
      // A completed task cannot be cancelled; nothing to do.
    }
    this.renderTask = null;
  }

  private async renderCurrentPage(): Promise<void> {
    const document = this.document;
    if (!document) {
      return;
    }

    const token = ++this.renderToken;
    this.cancelRenderTask();
    this.emitState(true);

    try {
      const page = await document.getPage(this.pageNumber);
      if (token !== this.renderToken) {
        return;
      }

      const baseViewport = page.getViewport({
        scale: 1,
        rotation: page.rotate,
      });
      let scale = this.scale;
      if (this.fitWidth) {
        const available = Math.max(this.stage.clientWidth - STAGE_PADDING, 240);
        scale = clamp(available / baseViewport.width, MIN_SCALE, MAX_SCALE);
      }

      const viewport = page.getViewport({ scale, rotation: page.rotate });
      this.scale = scale;
      this.currentViewport = viewport;

      const context = this.context;
      if (!context) {
        throw new Error("无法创建 Canvas 2D 上下文");
      }

      const outputScale = Math.min(window.devicePixelRatio || 1, 2.5);
      const cssWidth = viewport.width;
      const cssHeight = viewport.height;

      this.canvas.width = Math.max(1, Math.floor(cssWidth * outputScale));
      this.canvas.height = Math.max(1, Math.floor(cssHeight * outputScale));
      this.canvas.style.width = `${cssWidth}px`;
      this.canvas.style.height = `${cssHeight}px`;

      this.shell.style.width = `${cssWidth}px`;
      this.shell.style.height = `${cssHeight}px`;
      this.overlay.style.width = `${cssWidth}px`;
      this.overlay.style.height = `${cssHeight}px`;

      const renderTask = page.render({
        canvasContext: context,
        viewport,
        transform:
          outputScale !== 1
            ? [outputScale, 0, 0, outputScale, 0, 0]
            : undefined,
      });
      this.renderTask = renderTask;

      await renderTask.promise;
      if (token !== this.renderToken) {
        return;
      }

      this.renderTask = null;
      this.callbacks.onRender(this.pageNumber, viewport);
      this.emitState(false);
    } catch (error) {
      if (token !== this.renderToken || isRenderingCancelled(error)) {
        return;
      }
      this.renderTask = null;
      this.emitState(false);
      this.callbacks.onError(error);
    }
  }

  private emitState(rendering: boolean): void {
    this.callbacks.onStateChange({
      pageNumber: this.pageNumber,
      pageCount: this.pageCount,
      scale: this.currentViewport?.scale ?? this.scale,
      rendering,
      fitWidth: this.fitWidth,
      loaded: this.document !== null,
    });
  }
}
