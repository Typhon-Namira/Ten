import { useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { AIScoreSnapshot } from '../types'

const POLL_MS = 15_000

/** Backs the confidence-trend sparkline — a separate, slower-polled series rather than baked
 * into useLiveDashboard, since trend history doesn't need a 5s cadence like the rest of the
 * live panels. */
export function useAiScoreHistory(instrument: string, timeframe: string): AIScoreSnapshot[] {
  const [history, setHistory] = useState<AIScoreSnapshot[]>([])

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const value = await tenApi.aiScoreHistory(instrument, timeframe, 40)
        if (!cancelled) setHistory(value)
      } catch {
        // keep the last known-good history rather than blanking the sparkline
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [instrument, timeframe])

  return history
}
