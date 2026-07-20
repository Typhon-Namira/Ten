import { useCallback, useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import type { MarketIntelligence, PerformanceMetrics, PipelineStagesResponse, RejectionsResponse } from '../types'

const POLL_MS = 5_000

interface LiveDashboardState {
  stages: PipelineStagesResponse | null
  rejections: RejectionsResponse | null
  marketIntelligence: MarketIntelligence | null
  performance: PerformanceMetrics | null
  lastUpdated: Date | null
  errors: Record<string, string>
}

async function settle<T>(key: string, promise: Promise<T>, previous: T, errors: Record<string, string>): Promise<T> {
  try {
    const value = await promise
    delete errors[key]
    return value
  } catch (caught) {
    errors[key] = caught instanceof Error ? caught.message : 'request failed'
    return previous
  }
}

/** Polls the Part 3/4/5/7 aggregation endpoints every few seconds; the SSE stream (useEventStream)
 * covers the raw live log, this covers the composed "current state" views those events feed into. */
export function useLiveDashboard(): LiveDashboardState {
  const [stages, setStages] = useState<PipelineStagesResponse | null>(null)
  const [rejections, setRejections] = useState<RejectionsResponse | null>(null)
  const [marketIntelligence, setMarketIntelligence] = useState<MarketIntelligence | null>(null)
  const [performance, setPerformance] = useState<PerformanceMetrics | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const latest = useRef({ stages, rejections, marketIntelligence, performance })
  latest.current = { stages, rejections, marketIntelligence, performance }

  const refresh = useCallback(async () => {
    const current = latest.current
    const nextErrors: Record<string, string> = {}
    const [nextStages, nextRejections, nextIntelligence, nextPerformance] = await Promise.all([
      settle('stages', tenApi.pipelineStages(), current.stages, nextErrors),
      settle('rejections', tenApi.rejections(), current.rejections, nextErrors),
      settle('marketIntelligence', tenApi.marketIntelligence(), current.marketIntelligence, nextErrors),
      settle('performance', tenApi.performance(), current.performance, nextErrors),
    ])
    setStages(nextStages)
    setRejections(nextRejections)
    setMarketIntelligence(nextIntelligence)
    setPerformance(nextPerformance)
    setErrors(nextErrors)
    setLastUpdated(new Date())
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), POLL_MS)
    return () => window.clearInterval(timer)
  }, [refresh])

  return { stages, rejections, marketIntelligence, performance, lastUpdated, errors }
}
