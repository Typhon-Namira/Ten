import { useEffect, useState } from 'react'
import { fetchSafe } from '../services/api'

const POLL_MS = 5_000

export type EngineRecord = Record<string, unknown>

export interface EngineDetailData {
  health: EngineRecord | null
  metrics: EngineRecord | null
  state: EngineRecord | null
  loaded: boolean
}

/** Generic per-engine live data fetcher backing every engine detail page — polls that engine's
 * own `/health`, `/metrics`, and state-like endpoint (name varies per engine: `/state`,
 * `/snapshot`, `/diagnostics`) directly, loosely typed, rather than hand-writing a bespoke typed
 * client for each of the seven engines' large domain models. Both `symbol` and `instrument` query
 * params are sent on every request since engines disagree on which name they accept — FastAPI
 * silently ignores whichever one a given endpoint doesn't declare, so this is safe for all of them. */
export function useEngineDetail(basePath: string, statePath: string, instrument: string, timeframe: string): EngineDetailData {
  const [health, setHealth] = useState<EngineRecord | null>(null)
  const [metrics, setMetrics] = useState<EngineRecord | null>(null)
  const [state, setState] = useState<EngineRecord | null>(null)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    const query = `symbol=${encodeURIComponent(instrument)}&instrument=${encodeURIComponent(instrument)}&timeframe=${encodeURIComponent(timeframe)}`
    const refresh = async () => {
      const [nextHealth, nextMetrics, nextState] = await Promise.all([
        fetchSafe<EngineRecord>(`${basePath}/health`),
        fetchSafe<EngineRecord>(`${basePath}/metrics`),
        fetchSafe<EngineRecord>(`${basePath}${statePath}?${query}`),
      ])
      if (cancelled) return
      setHealth(nextHealth)
      setMetrics(nextMetrics)
      setState(nextState)
      setLoaded(true)
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [basePath, statePath, instrument, timeframe])

  return { health, metrics, state, loaded }
}
