import { useCallback, useRef, useState } from 'react'
import type { ExplainResponse } from '../types'

/** Drives one on-demand `/api/v1/explain/*` call — never auto-polled like the rest of the
 * dashboard, since each call is a real LLM round trip. `fetcher` is read from a ref so callers can
 * pass a fresh closure every render without retriggering anything themselves. */
export function useExplain(fetcher: () => Promise<ExplainResponse>) {
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher
  const [data, setData] = useState<ExplainResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetcherRef.current())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'explanation request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, run }
}
