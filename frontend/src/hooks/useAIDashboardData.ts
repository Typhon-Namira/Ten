import { useCallback, useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import type { AIReasoningDashboard, DashboardAggregate, MarketIntelligence, QuantCalibrationReport, QuantForecastResult } from '../types'

export interface AIDashboardData {
  intelligence: MarketIntelligence | null
  quant: QuantForecastResult | null
  calibration: QuantCalibrationReport | null
  reasoning: AIReasoningDashboard | null
  aggregate: DashboardAggregate | null
  loading: boolean
  stale: boolean
  errors: Record<string, string>
  lastUpdated: Date | null
  refresh: () => Promise<void>
}

const POLL_MS = 15_000
const HIDDEN_POLL_MS = 60_000
const MAX_BACKOFF_MS = 120_000

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Backend unavailable'
}

export function useAIDashboardData(instrument: string, timeframe: string): AIDashboardData {
  const [intelligence, setIntelligence] = useState<MarketIntelligence | null>(null)
  const [quant, setQuant] = useState<QuantForecastResult | null>(null)
  const [calibration, setCalibration] = useState<QuantCalibrationReport | null>(null)
  const [reasoning, setReasoning] = useState<AIReasoningDashboard | null>(null)
  const [aggregate, setAggregate] = useState<DashboardAggregate | null>(null)
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
      ] as const)
      const nextErrors: Record<string, string> = {}
      if (results[0].status === 'rejected') nextErrors.market = message(results[0].reason)
      if (results[1].status === 'rejected') nextErrors.dashboard = message(results[1].reason)
      if (results[0].status === 'fulfilled') setIntelligence(results[0].value)
      if (results[1].status === 'fulfilled') {
        const value = results[1].value
        setAggregate(value)
        setQuant(value.stages.quant_forecast.data)
        setCalibration(value.calibration.data)
        setReasoning(value.reasoning)
      }
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
  return { intelligence, quant, calibration, reasoning, aggregate, loading, stale, errors, lastUpdated, refresh }
}
