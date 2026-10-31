import { emptySelection } from "./selection";
import type { HighlightSelection } from "./selection";

/** Which interaction currently owns the single PDF highlight overlay. */
export type HighlightSource = "anchor" | "chunk" | "qa";

export interface HighlightState {
  selection: HighlightSelection;
  source: HighlightSource | null;
}

export function emptyHighlightState(): HighlightState {
  return { selection: emptySelection(), source: null };
}

export function highlightFor(
  source: HighlightSource,
  selection: HighlightSelection
): HighlightState {
  return { selection, source };
}

/**
 * Clear the overlay. When `source` is provided, only that source may be
 * cleared: QA cleanup must not remove a literal-search or chunk-evidence
 * highlight.
 */
export function clearHighlightFor(
  state: HighlightState,
  source?: HighlightSource
): HighlightState {
  if (source !== undefined && state.source !== source) {
    return state;
  }
  return emptyHighlightState();
}
