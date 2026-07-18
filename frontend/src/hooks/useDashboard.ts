import { useCallback, useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { AIScoreSnapshot, EngineStatus, MarketStatus, Signal } from '../types'

interface DashboardState {
  signals: Signal[]
  engines: EngineStatus[]
  market: MarketStatus | null
  aiScore: AIScoreSnapshot | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

export function useDashboard(): DashboardState {
  const [signals, setSignals] = useState<Signal[]>([])
  const [engines, setEngines] = useState<EngineStatus[]>([])
  const [market, setMarket] = useState<MarketStatus | null>(null)
  const [aiScore, setAIScore] = useState<AIScoreSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const [nextSignals, nextEngines, nextMarket, nextAIScore] = await Promise.all([
        tenApi.signals(),
        tenApi.engines(),
        tenApi.market(),
        tenApi.latestAIScore(),
      ])
      setSignals(nextSignals)
      setEngines(nextEngines)
      setMarket(nextMarket)
      setAIScore(nextAIScore)
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

  return { signals, engines, market, aiScore, loading, error, refresh }
}

