/**
 * Monotonic request tokens for asynchronous UI work.
 *
 * A gate never cancels the underlying network request; it only tells the
 * caller whether the response still belongs to the newest interaction. This
 * keeps older chunk searches or evidence locations from overwriting the state
 * produced by a newer click or a document switch.
 */
export interface RequestGate {
  /** Start a new request and invalidate every older one. */
  begin(): number;
  /** Whether the given request is still the newest request. */
  isCurrent(token: number): boolean;
}

export function createRequestGate(): RequestGate {
  let current = 0;
  return {
    begin(): number {
      current += 1;
      return current;
    },
    isCurrent(token: number): boolean {
      return token === current;
    },
  };
}
