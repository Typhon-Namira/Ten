import { useEffect, useState } from 'react'
import type { PerformanceMetrics } from '../types'

const MAX_POINTS = 60

export interface PerformanceHistoryPoint {
  pipelineLatency: number | null
  providerLatency: number | null
  queueLength: number | null
}

/** `/performance` is a live snapshot, not a time series — the backend has no historical-trend
 * buffer for it. Rather than add one server-side, this accumulates each poll's snapshot into a
 * bounded client-side rolling window via an effect (not during render, to stay StrictMode-safe),
 * which is exactly what a "trend chart" needs without a backend change for a purely
 * presentational rolling history. */
export function usePerformanceHistory(snapshot: PerformanceMetrics | null): PerformanceHistoryPoint[] {
  const [history, setHistory] = useState<PerformanceHistoryPoint[]>([])

  useEffect(() => {
    if (!snapshot) return
    let active = true
    queueMicrotask(() => {
      if (!active) return
      setHistory((previous) =>
        [
          ...previous,
          {
            pipelineLatency: snapshot.pipeline_latency_ms ?? snapshot.pipeline_in_flight_ms,
            providerLatency: snapshot.provider.last_latency_ms,
            queueLength: snapshot.queue_length,
          },
        ].slice(-MAX_POINTS),
      )
    })
    // Re-runs whenever the snapshot object identity changes, which happens on every poll cycle
    // in useLiveDashboard (each refresh sets new state) — no extra dedupe key needed.
    return () => {
      active = false
    }
  }, [snapshot])

  return history
}
