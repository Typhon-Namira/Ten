import type { AIScoreSnapshot, EngineStatus, MarketStatus, OperationalSignal, ReplaySessionOverview, Signal, SignalDecisionSnapshot, SystemDiagnostics } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, '') ?? ''

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`)
  if (!response.ok) {
    throw new Error(`TEN API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

async function requestOptional<T>(path: string): Promise<T | null> {
  const response = await fetch(`${API_BASE_URL}${path}`)
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`TEN API request failed (${response.status})`)
  return response.json() as Promise<T>
}

export const tenApi = {
  signals: () => request<Signal[]>('/signals?limit=20'),
  engines: () => request<EngineStatus[]>('/engines/status'),
  market: () => request<MarketStatus>('/market/status'),
  diagnostics: () => request<SystemDiagnostics>('/api/v1/system/diagnostics'),
  latestAIScore: () => requestOptional<AIScoreSnapshot>('/ai-scoring/latest?instrument=XAUUSD&timeframe=M15'),
  latestSignalDecision: () => requestOptional<SignalDecisionSnapshot>('/signal-decisions/latest?instrument=XAUUSD&timeframe=M15'),
  latestOperationalSignal: () => requestOptional<OperationalSignal>('/integration/signals/latest?instrument=XAUUSD&timeframe=M15'),
  replays: () => request<ReplaySessionOverview[]>('/replays?limit=5'),
}
