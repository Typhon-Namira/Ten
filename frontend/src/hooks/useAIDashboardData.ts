import { useCallback, useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { AIReasoningDashboard, MarketIntelligence, QuantCalibrationReport, QuantForecastResult } from '../types'

export interface AIDashboardData {
  intelligence: MarketIntelligence | null
  quant: QuantForecastResult | null
  calibration: QuantCalibrationReport | null
  reasoning: AIReasoningDashboard | null
  loading: boolean
  stale: boolean
  errors: Record<string, string>
  lastUpdated: Date | null
  refresh: () => Promise<void>
}

const POLL_MS = 5_000

function message(error: unknown): string {
  return error instanceof Error ? error.message : 'Backend unavailable'
}

export function useAIDashboardData(instrument: string, timeframe: string): AIDashboardData {
  const [intelligence, setIntelligence] = useState<MarketIntelligence | null>(null)
  const [quant, setQuant] = useState<QuantForecastResult | null>(null)
  const [calibration, setCalibration] = useState<QuantCalibrationReport | null>(null)
  const [reasoning, setReasoning] = useState<AIReasoningDashboard | null>(null)
  const [loading, setLoading] = useState(true)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const refresh = useCallback(async () => {
    const results = await Promise.allSettled([
      tenApi.marketIntelligence(instrument, timeframe),
      tenApi.latestQuantForecast(instrument),
      tenApi.latestQuantCalibration(),
      tenApi.latestAIReasoning(instrument),
    ] as const)
    const nextErrors: Record<string, string> = {}
    const keys = ['market', 'quantitative forecast', 'calibration', 'AI reasoning'] as const
    results.forEach((result, index) => {
      if (result.status === 'rejected') nextErrors[keys[index]] = message(result.reason)
    })
    if (results[0].status === 'fulfilled') setIntelligence(results[0].value)
    if (results[1].status === 'fulfilled') setQuant(results[1].value)
    if (results[2].status === 'fulfilled') setCalibration(results[2].value)
    if (results[3].status === 'fulfilled') setReasoning(results[3].value)
    setErrors(nextErrors)
    setLastUpdated(new Date())
    setLoading(false)
  }, [instrument, timeframe])

  useEffect(() => {
    const initial = window.setTimeout(() => void refresh(), 0)
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => {
      window.clearTimeout(initial)
      window.clearInterval(timer)
    }
  }, [refresh])

  const stale = Boolean(
    intelligence?.diagnostics.some(item => item.freshness === 'stale')
    || intelligence?.latest_candle_timestamp == null,
  )
  return { intelligence, quant, calibration, reasoning, loading, stale, errors, lastUpdated, refresh }
}
