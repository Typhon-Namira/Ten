import type { ApiError, ApiErrorKind } from './apiError'

/** Process-local, no-backend-call registry of "how did the last fetch for this data source go" —
 * feeds the always-visible diagnostics bar (components/DiagnosticsBar.tsx). Every hook that talks
 * to the backend calls `recordFetchOutcome()` after each attempt; nothing here issues a network
 * request of its own. */
export interface FetchRecord {
  source: string
  label: string
  lastAttemptAt: Date
  lastSuccessAt: Date | null
  lastError: { kind: ApiErrorKind; status: number | null; message: string } | null
}

type Listener = () => void

const records = new Map<string, FetchRecord>()
const listeners = new Set<Listener>()

const LABELS: Record<string, string> = {
  'market-intelligence': 'Market intelligence',
  dashboard: 'AI dashboard',
  diagnostics: 'System diagnostics',
}

function label(source: string): string {
  return LABELS[source] ?? source.replace(/-/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
}

export function recordFetchOutcome(source: string, outcome: { ok: true } | { ok: false; error: ApiError }): void {
  const now = new Date()
  const previous = records.get(source)
  records.set(source, {
    source,
    label: label(source),
    lastAttemptAt: now,
    lastSuccessAt: outcome.ok ? now : (previous?.lastSuccessAt ?? null),
    lastError: outcome.ok ? null : { kind: outcome.error.kind, status: outcome.error.status, message: outcome.error.message },
  })
  for (const listener of listeners) listener()
}

export function subscribeToDiagnosticsFeed(listener: Listener): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

let cachedSnapshot: FetchRecord[] = []

export function getDiagnosticsFeedSnapshot(): FetchRecord[] {
  const next = Array.from(records.values()).sort((a, b) => a.source.localeCompare(b.source))
  // useSyncExternalStore requires a stable reference when nothing changed, or it re-renders forever.
  if (next.length === cachedSnapshot.length && next.every((item, index) => item === cachedSnapshot[index])) {
    return cachedSnapshot
  }
  cachedSnapshot = next
  return next
}
