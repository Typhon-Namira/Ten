import type { AIScoreSnapshot, EngineStatus, MarketStatus, Signal, SignalDecisionSnapshot } from '../types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${API_URL}${path}`)
  if (!response.ok) {
    throw new Error(`TEN API request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

async function requestOptional<T>(path: string): Promise<T | null> {
  const response = await fetch(`${API_URL}${path}`)
  if (response.status === 404) return null
  if (!response.ok) throw new Error(`TEN API request failed (${response.status})`)
  return response.json() as Promise<T>
}

export const tenApi = {
  signals: () => request<Signal[]>('/signals?limit=20'),
  engines: () => request<EngineStatus[]>('/engines/status'),
  market: () => request<MarketStatus>('/market/status'),
  latestAIScore: () => requestOptional<AIScoreSnapshot>('/ai-scoring/latest?instrument=XAUUSD&timeframe=M15'),
  latestSignalDecision: () => requestOptional<SignalDecisionSnapshot>('/signal-decisions/latest?instrument=XAUUSD&timeframe=M15'),
}

