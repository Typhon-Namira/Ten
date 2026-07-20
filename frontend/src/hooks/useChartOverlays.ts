import { useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { ChartOverlays } from '../types'

const POLL_MS = 5_000

/** Same instrument/timeframe pair as every other live hook (see useActiveSelection) — the chart
 * must never silently look at a different candle series than the rest of the dashboard. */
export function useChartOverlays(instrument: string, timeframe: string): { data: ChartOverlays | null; error: string | null } {
  const [data, setData] = useState<ChartOverlays | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const value = await tenApi.chartOverlays(instrument, timeframe)
        if (!cancelled) {
          setData(value)
          setError(null)
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : 'request failed')
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [instrument, timeframe])

  return { data, error }
}
