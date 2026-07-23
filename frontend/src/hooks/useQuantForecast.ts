import { useEffect, useState } from 'react'
import { tenApi } from '../services/api'
import type { QuantCalibrationReport, QuantForecastOutcome, QuantForecastResult } from '../types'

export function useQuantForecast(instrument: string): { forecast: QuantForecastResult | null; calibration: QuantCalibrationReport | null; outcomes: QuantForecastOutcome[] } {
  const [forecast, setForecast] = useState<QuantForecastResult | null>(null)
  const [calibration, setCalibration] = useState<QuantCalibrationReport | null>(null)
  const [outcomes, setOutcomes] = useState<QuantForecastOutcome[]>([])

  useEffect(() => {
    let cancelled = false
    const refresh = async () => {
      try {
        const [forecastValue, calibrationValue] = await Promise.all([
          tenApi.latestQuantForecast(instrument),
          tenApi.latestQuantCalibration(),
        ])
        const outcomeValues = forecastValue ? await tenApi.quantForecastOutcomes(forecastValue.result_id) : []
        if (!cancelled) {
          setForecast(forecastValue)
          setCalibration(calibrationValue)
          setOutcomes(outcomeValues)
        }
      } catch {
        // Preserve the last persisted shadow result during transient observability failures.
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 15_000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [instrument])

  return { forecast, calibration, outcomes }
}
