import { useCallback, useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { EngineStatus, MarketStatus, Signal } from '../types'

interface DashboardState {
  signals: Signal[]
  engines: EngineStatus[]
  market: MarketStatus | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

export function useDashboard(): DashboardState {
  const [signals, setSignals] = useState<Signal[]>([])
  const [engines, setEngines] = useState<EngineStatus[]>([])
  const [market, setMarket] = useState<MarketStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [nextSignals, nextEngines, nextMarket] = await Promise.all([
        tenApi.signals(),
        tenApi.engines(),
        tenApi.market(),
      ])
      setSignals(nextSignals)
      setEngines(nextEngines)
      setMarket(nextMarket)
      setError(null)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Unable to reach TEN API')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 30_000)
    return () => window.clearInterval(timer)
  }, [refresh])

  return { signals, engines, market, loading, error, refresh }
}

