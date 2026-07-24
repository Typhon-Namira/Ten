import { useCallback, useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import { describeApiError, toApiError } from '../lib/apiError'
import { recordFetchOutcome } from '../lib/diagnosticsFeed'
import type { AIReasoningDashboard, DashboardAggregate, DashboardSystemStatus, MarketIntelligence, QuantCalibrationReport, QuantForecastResult } from '../types'

export interface AIDashboardData {
  intelligence: MarketIntelligence | null
  quant: QuantForecastResult | null
  calibration: QuantCalibrationReport | null
  reasoning: AIReasoningDashboard | null
  aggregate: DashboardAggregate | null
  systemStatus: DashboardSystemStatus | null
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
  const [calibration, setCalibration] = useState<QuantCalibrationReport | null>(null)
  const [reasoning, setReasoning] = useState<AIReasoningDashboard | null>(null)
  const [aggregate, setAggregate] = useState<DashboardAggregate | null>(null)
  const [systemStatus, setSystemStatus] = useState<DashboardSystemStatus | null>(null)
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
        tenApi.dashboardLatest(instrument),
        tenApi.dashboardSystemStatus(instrument),
      ] as const)
      const nextErrors: Record<string, string> = {}
      const sources = ['market-intelligence', 'dashboard', 'dashboard-system'] as const
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
        setAggregate(value)
        setQuant(value.stages.quant_forecast.data)
        setCalibration(value.calibration.data)
        setReasoning(value.reasoning)
      }
      if (results[2].status === 'fulfilled') setSystemStatus(results[2].value)
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
  return { intelligence, quant, calibration, reasoning, aggregate, systemStatus, loading, stale, errors, lastUpdated, refresh }
}
