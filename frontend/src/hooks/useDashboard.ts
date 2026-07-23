import { useCallback, useEffect, useRef, useState } from 'react'
import { tenApi } from '../services/api'
import type { AIScoreSnapshot, EngineStatus, MarketStatus, OperationalSignal, ReplaySessionOverview, SignalDecisionSnapshot, SystemDiagnostics } from '../types'

interface DashboardState {
  signals: OperationalSignal[]
  engines: EngineStatus[]
  market: MarketStatus | null
  aiScore: AIScoreSnapshot | null
  signalDecision: SignalDecisionSnapshot | null
  operationalSignal: OperationalSignal | null
  replays: ReplaySessionOverview[]
  diagnostics: SystemDiagnostics | null
  loading: boolean
  error: string | null
  refresh: () => Promise<void>
}

/** Settles a fetch without letting one failing source blank out an already-loaded value. */
async function settle<T>(promise: Promise<T>, previous: T): Promise<{ value: T; error: string | null }> {
  try {
    return { value: await promise, error: null }
  } catch (caught) {
    return { value: previous, error: caught instanceof Error ? caught.message : 'request failed' }
  }
}

export function useDashboard(instrument: string, timeframe: string): DashboardState {
  const [signals, setSignals] = useState<OperationalSignal[]>([])
  const [engines, setEngines] = useState<EngineStatus[]>([])
  const [market, setMarket] = useState<MarketStatus | null>(null)
  const [aiScore, setAIScore] = useState<AIScoreSnapshot | null>(null)
  const [signalDecision, setSignalDecision] = useState<SignalDecisionSnapshot | null>(null)
  const [operationalSignal, setOperationalSignal] = useState<OperationalSignal | null>(null)
  const [replays, setReplays] = useState<ReplaySessionOverview[]>([])
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const latest = useRef({ signals, engines, market, aiScore, signalDecision, operationalSignal, replays, diagnostics })

  useEffect(() => {
    latest.current = { signals, engines, market, aiScore, signalDecision, operationalSignal, replays, diagnostics }
  }, [signals, engines, market, aiScore, signalDecision, operationalSignal, replays, diagnostics])

  const refresh = useCallback(async () => {
    setLoading(true)
    const current = latest.current
    const [nextSignals, nextEngines, nextMarket, nextAIScore, nextDecision, nextOperational, nextReplays, nextDiagnostics] = await Promise.all([
      settle(tenApi.signals(), current.signals),
      settle(tenApi.engines(), current.engines),
      settle(tenApi.market(), current.market),
      settle(tenApi.latestAIScore(instrument, timeframe), current.aiScore),
      settle(tenApi.latestSignalDecision(instrument, timeframe), current.signalDecision),
      settle(tenApi.latestOperationalSignal(instrument, timeframe), current.operationalSignal),
      settle(tenApi.replays(), current.replays),
      settle(tenApi.diagnostics(), current.diagnostics),
    ])
    setSignals(nextSignals.value)
    setEngines(nextEngines.value)
    setMarket(nextMarket.value)
    setAIScore(nextAIScore.value)
    setSignalDecision(nextDecision.value)
    setOperationalSignal(nextOperational.value)
    setReplays(nextReplays.value)
    setDiagnostics(nextDiagnostics.value)
    const firstError = [nextSignals, nextEngines, nextMarket, nextAIScore, nextDecision, nextOperational, nextReplays, nextDiagnostics].find((item) => item.error)?.error
    setError(firstError ?? null)
    setLoading(false)
  }, [instrument, timeframe])

  useEffect(() => {
    let active = true
    queueMicrotask(() => {
      if (active) void refresh()
    })
    const timer = window.setInterval(() => void refresh(), 30_000)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [refresh])

  return { signals, engines, market, aiScore, signalDecision, operationalSignal, replays, diagnostics, loading, error, refresh }
}
