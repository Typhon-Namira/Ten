/** Distinguishes *why* a request didn't produce data — collapsing these into one generic "failed"
 * string is exactly what made a real backend failure (Issue 1: OpenRouter auth) indistinguishable
 * from a frontend contract mismatch or a request that simply never completed. */
export type ApiErrorKind = 'timeout' | 'network' | 'http' | 'parse'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number | null

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
  }
}

/** Best-effort classification for an error this module didn't originate (e.g. a JSON parse
 * exception thrown outside `request()`, or a plain `Error` from older call sites). */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  if (error instanceof DOMException && error.name === 'AbortError') return new ApiError('timeout', 'Request timed out')
  if (error instanceof TypeError) return new ApiError('network', error.message || 'Network request failed')
  if (error instanceof Error) return new ApiError('http', error.message)
  return new ApiError('network', 'Unknown request failure')
}

/** Short, human-facing label for the diagnostics bar / inline error banners. */
export function describeApiError(error: ApiError): string {
  switch (error.kind) {
    case 'timeout':
      return 'Timed out — the backend did not respond in time'
    case 'network':
      return `Network error — ${error.message}`
    case 'parse':
      return `Unexpected response shape — ${error.message}`
    case 'http':
      return error.status ? `Backend returned HTTP ${error.status} — ${error.message}` : error.message
  }
}
