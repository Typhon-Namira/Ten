import { useCallback, useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import { describeApiError, toApiError } from '../lib/apiError'
import { recordFetchOutcome } from '../lib/diagnosticsFeed'
import type { AnalysisHistoryPage, AnalysisSignalPage, DashboardSystemStatus, LatestCompletedCycle, MarketIntelligence, QuantForecastResult } from '../types'

export interface AIDashboardData {
  intelligence: MarketIntelligence | null
  quant: QuantForecastResult | null
  systemStatus: DashboardSystemStatus | null
  latestCycle: LatestCompletedCycle | null
  signalHistory: AnalysisSignalPage | null
  analysisHistory: AnalysisHistoryPage | null
  loading: boolean
  stale: boolean
  errors: Record<string, string>
  lastUpdated: Date | null
  refresh: () => Promise<void>
}

const POLL_MS = 15_000
const HIDDEN_POLL_MS = 60_000
const MAX_BACKOFF_MS = 120_000

export function useAIDashboardData(instrument: string, timeframe: string): AIDashboardData {
  const [intelligence, setIntelligence] = useState<MarketIntelligence | null>(null)
  const [quant, setQuant] = useState<QuantForecastResult | null>(null)
  const [systemStatus, setSystemStatus] = useState<DashboardSystemStatus | null>(null)
  const [latestCycle, setLatestCycle] = useState<LatestCompletedCycle | null>(null)
  const [signalHistory, setSignalHistory] = useState<AnalysisSignalPage | null>(null)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const inFlight = useRef<Promise<void> | null>(null)
  const failureCount = useRef(0)
  const refresh = useCallback(async () => {
    if (inFlight.current) return inFlight.current
    const operation = (async () => {
      const results = await Promise.allSettled([
        tenApi.marketIntelligence(instrument, timeframe),
        tenApi.dashboardLatestCycle(instrument, timeframe),
        tenApi.dashboardSystemStatus(instrument),
        tenApi.dashboardSignals(instrument, timeframe),
        tenApi.dashboardAnalyses(instrument, timeframe),
      ] as const)
      const nextErrors: Record<string, string> = {}
      const sources = ['market-intelligence', 'latest-cycle', 'dashboard-system', 'signal-history', 'analysis-history'] as const
      results.forEach((result, index) => {
        if (result.status === 'rejected') {
          const error = toApiError(result.reason)
          nextErrors[sources[index]] = describeApiError(error)
          recordFetchOutcome(sources[index], { ok: false, error })
        } else {
          recordFetchOutcome(sources[index], { ok: true })
        }
      })
      if (results[0].status === 'fulfilled') setIntelligence(results[0].value)
      if (results[1].status === 'fulfilled') {
        const value = results[1].value
        setLatestCycle(value)
        setQuant(value.quant_forecast ?? null)
      }
      if (results[2].status === 'fulfilled') setSystemStatus(results[2].value)
      if (results[3].status === 'fulfilled') setSignalHistory(results[3].value)
      if (results[4].status === 'fulfilled') setAnalysisHistory(results[4].value)
      failureCount.current = Object.keys(nextErrors).length ? failureCount.current + 1 : 0
      setErrors(nextErrors)
      setLastUpdated(new Date())
      setLoading(false)
    })()
    inFlight.current = operation
    try {
      await operation
    } finally {
      inFlight.current = null
    }
  }, [instrument, timeframe])

  useEffect(() => {
    let active = true
    let timer: number | undefined
    const poll = async () => {
      await refresh()
      if (!active) return
      const base = document.hidden ? HIDDEN_POLL_MS : POLL_MS
      const delay = Math.min(base * (2 ** failureCount.current), MAX_BACKOFF_MS)
      timer = window.setTimeout(() => void poll(), delay)
    }
    void poll()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [refresh])

  const stale = Boolean(
    intelligence?.diagnostics.some(item => item.freshness === 'stale')
    || intelligence?.latest_candle_timestamp == null,
  )
  return { intelligence, quant, systemStatus, latestCycle, signalHistory, analysisHistory, loading, stale, errors, lastUpdated, refresh }
}
