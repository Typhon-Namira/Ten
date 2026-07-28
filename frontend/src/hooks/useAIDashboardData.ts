import { useCallback, useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import { describeApiError, toApiError } from '../lib/apiError'
import { recordFetchOutcome } from '../lib/diagnosticsFeed'
import type {
  AnalysisHistoryPage,
  AnalysisSignalPage,
  DashboardSystemStatus,
  LatestCompletedCycle,
  MarketIntelligence,
  QuantForecastResult,
} from '../types'

export interface AIDashboardData {
  intelligence: MarketIntelligence | null
  quant: QuantForecastResult | null
  systemStatus: DashboardSystemStatus | null
  latestCycle: LatestCompletedCycle | null
  signalHistory: AnalysisSignalPage | null
  analysisHistory: AnalysisHistoryPage | null
  loading: boolean
  refreshing: boolean
  stale: boolean
  errors: Record<string, string>
  lastChecked: Date | null
  refresh: () => Promise<void>
}

const configuredPollMs = Number(import.meta.env.VITE_DASHBOARD_POLL_INTERVAL_MS ?? 3_000)
export const DASHBOARD_POLL_INTERVAL_MS = Number.isFinite(configuredPollMs)
  ? Math.max(1_000, configuredPollMs)
  : 3_000
const HIDDEN_POLL_MS = Math.max(60_000, DASHBOARD_POLL_INTERVAL_MS)
const ANCILLARY_POLL_MS = 15_000
const MAX_BACKOFF_MS = 120_000

function timestamp(value: string | null | undefined): number {
  if (!value) return 0
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? 0 : parsed
}

/** A stable total ordering prevents a late response from replacing a newer coherent cycle. */
export function compareCompletedCycles(
  candidate: LatestCompletedCycle,
  current: LatestCompletedCycle,
): number {
  if (candidate.status !== current.status) {
    return candidate.status === 'completed' ? 1 : -1
  }
  const candidateOrder = [
    timestamp(candidate.analysis_timestamp),
    timestamp(candidate.decision_timestamp),
    timestamp(candidate.updated_at),
  ]
  const currentOrder = [
    timestamp(current.analysis_timestamp),
    timestamp(current.decision_timestamp),
    timestamp(current.updated_at),
  ]
  for (let index = 0; index < candidateOrder.length; index += 1) {
    if (candidateOrder[index] !== currentOrder[index]) {
      return candidateOrder[index] > currentOrder[index] ? 1 : -1
    }
  }
  return (candidate.cycle_id ?? '').localeCompare(current.cycle_id ?? '')
}

function developmentLog(event: string, details: Record<string, unknown>) {
  if (import.meta.env.DEV) console.info(event, details)
}

export function useAIDashboardData(instrument: string, timeframe: string): AIDashboardData {
  const [intelligence, setIntelligence] = useState<MarketIntelligence | null>(null)
  const [quant, setQuant] = useState<QuantForecastResult | null>(null)
  const [systemStatus, setSystemStatus] = useState<DashboardSystemStatus | null>(null)
  const [latestCycle, setLatestCycle] = useState<LatestCompletedCycle | null>(null)
  const [signalHistory, setSignalHistory] = useState<AnalysisSignalPage | null>(null)
  const [analysisHistory, setAnalysisHistory] = useState<AnalysisHistoryPage | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [lastChecked, setLastChecked] = useState<Date | null>(null)
  const latestCycleRef = useRef<LatestCompletedCycle | null>(null)
  const latestCycleInFlight = useRef<Promise<void> | null>(null)
  const latestCycleController = useRef<AbortController | null>(null)
  const ancillaryController = useRef<AbortController | null>(null)
  const ancillaryInFlight = useRef<Promise<void> | null>(null)
  const mounted = useRef(false)
  const latestFailureCount = useRef(0)

  useEffect(() => {
    latestCycleRef.current = latestCycle
  }, [latestCycle])

  const refreshLatestCycle = useCallback(async () => {
    if (latestCycleInFlight.current) return latestCycleInFlight.current
    const controller = new AbortController()
    latestCycleController.current = controller
    const operation = (async () => {
      if (mounted.current) setRefreshing(true)
      try {
        const value = await tenApi.dashboardLatestCycle(instrument, controller.signal)
        if (
          !mounted.current
          || controller.signal.aborted
          || latestCycleController.current !== controller
        ) return
        const current = latestCycleRef.current
        developmentLog('dashboard.cycle.received', {
          instrument,
          cycle_id: value.cycle_id,
          cycle_version: value.cycle_version,
        })
        if (current == null || compareCompletedCycles(value, current) >= 0) {
          if (current?.cycle_id !== value.cycle_id) {
            developmentLog('dashboard.cycle.replaced', {
              previous_cycle_id: current?.cycle_id ?? null,
              cycle_id: value.cycle_id,
            })
          }
          latestCycleRef.current = value
          setLatestCycle(value)
          setQuant(value.quant_forecast ?? null)
        } else {
          developmentLog('dashboard.cycle.ignored_stale_response', {
            received_cycle_id: value.cycle_id,
            current_cycle_id: current.cycle_id,
          })
        }
        latestFailureCount.current = 0
        setLastChecked(new Date())
        setErrors(previous => {
          const next = { ...previous }
          delete next['latest-cycle']
          return next
        })
        recordFetchOutcome('latest-cycle', { ok: true })
      } catch (caught) {
        if (
          controller.signal.aborted
          || !mounted.current
          || latestCycleController.current !== controller
        ) return
        const error = toApiError(caught)
        latestFailureCount.current += 1
        setErrors(previous => ({ ...previous, 'latest-cycle': describeApiError(error) }))
        recordFetchOutcome('latest-cycle', { ok: false, error })
        developmentLog('dashboard.poll.failed', {
          instrument,
          error: describeApiError(error),
        })
      } finally {
        if (mounted.current && latestCycleController.current === controller) {
          setRefreshing(false)
          setLoading(false)
        }
      }
    })()
    latestCycleInFlight.current = operation
    try {
      await operation
    } finally {
      if (latestCycleInFlight.current === operation) latestCycleInFlight.current = null
      if (latestCycleController.current === controller) latestCycleController.current = null
    }
  }, [instrument])

  const refreshAncillary = useCallback(async () => {
    if (ancillaryInFlight.current) return ancillaryInFlight.current
    const controller = new AbortController()
    ancillaryController.current = controller
    const operation = (async () => {
      const results = await Promise.allSettled([
        tenApi.marketIntelligence(instrument, timeframe, controller.signal),
        tenApi.dashboardSystemStatus(instrument, controller.signal),
        tenApi.dashboardSignals(instrument, 0, 50, controller.signal),
        tenApi.dashboardAnalyses(instrument, 0, 50, controller.signal),
      ] as const)
      if (
        !mounted.current
        || controller.signal.aborted
        || ancillaryController.current !== controller
      ) return
      const sources = ['market-intelligence', 'dashboard-system', 'signal-history', 'analysis-history'] as const
      const nextErrors: Record<string, string> = {}
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
      if (results[1].status === 'fulfilled') setSystemStatus(results[1].value)
      if (results[2].status === 'fulfilled') setSignalHistory(results[2].value)
      if (results[3].status === 'fulfilled') setAnalysisHistory(results[3].value)
      setErrors(previous => {
        const next = { ...previous }
        for (const source of sources) delete next[source]
        return { ...next, ...nextErrors }
      })
    })()
    ancillaryInFlight.current = operation
    try {
      await operation
    } finally {
      if (ancillaryInFlight.current === operation) ancillaryInFlight.current = null
      if (ancillaryController.current === controller) ancillaryController.current = null
    }
  }, [instrument, timeframe])

  const refresh = useCallback(
    async () => {
      await Promise.all([refreshLatestCycle(), refreshAncillary()])
    },
    [refreshAncillary, refreshLatestCycle],
  )

  useEffect(() => {
    mounted.current = true
    latestCycleRef.current = null
    setLatestCycle(null)
    setQuant(null)
    setIntelligence(null)
    setSystemStatus(null)
    setSignalHistory(null)
    setAnalysisHistory(null)
    setLoading(true)
    let active = true
    let timer: number | undefined
    const schedule = (delay: number) => {
      if (timer !== undefined) window.clearTimeout(timer)
      if (active) timer = window.setTimeout(() => void poll(), delay)
    }
    const poll = async () => {
      await refreshLatestCycle()
      if (!active) return
      const base = document.hidden ? HIDDEN_POLL_MS : DASHBOARD_POLL_INTERVAL_MS
      schedule(Math.min(base * (2 ** latestFailureCount.current), MAX_BACKOFF_MS))
    }
    const visibilityChanged = () => {
      if (!document.hidden) schedule(0)
    }
    document.addEventListener('visibilitychange', visibilityChanged)
    void poll()
    void refreshAncillary()
    const ancillaryTimer = window.setInterval(
      () => void refreshAncillary(),
      ANCILLARY_POLL_MS,
    )
    return () => {
      active = false
      mounted.current = false
      if (timer !== undefined) window.clearTimeout(timer)
      window.clearInterval(ancillaryTimer)
      document.removeEventListener('visibilitychange', visibilityChanged)
      latestCycleController.current?.abort()
      ancillaryController.current?.abort()
    }
  }, [refreshAncillary, refreshLatestCycle])

  const stale = Boolean(
    intelligence?.diagnostics.some(item => item.freshness === 'stale')
    || intelligence?.latest_candle_timestamp == null,
  )
  return {
    intelligence,
    quant,
    systemStatus,
    latestCycle,
    signalHistory,
    analysisHistory,
    loading,
    refreshing,
    stale,
    errors,
    lastChecked,
    refresh,
  }
}
