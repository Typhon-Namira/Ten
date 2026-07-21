import type { ActiveSelection, AIScoreSnapshot, ChartOverlays, EngineStatus, MarketIntelligence, MarketStatus, OperationalSignal, PerformanceMetrics, PipelineStagesResponse, RejectionsResponse, ReplaySessionOverview, SignalDecisionSnapshot, SystemDiagnostics } from '../types'

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

/** `instrument`/`timeframe` must always come from `tenApi.selection()` (see useActiveSelection) —
 * never a literal — so every endpoint below is queried about the same candle series. */
function scoped(instrument: string, timeframe: string): string {
  return `instrument=${encodeURIComponent(instrument)}&timeframe=${encodeURIComponent(timeframe)}`
}

/** Never throws — resolves to `null` on any failure. Used by the generic engine-detail pages,
 * which fetch several loosely-typed endpoints in parallel and must not let one failing source
 * blank out the others. */
export async function fetchSafe<T>(path: string): Promise<T | null> {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`)
    if (!response.ok) return null
    return (await response.json()) as T
  } catch {
    return null
  }
}

export const tenApi = {
  selection: () => request<ActiveSelection>('/api/v1/system/selection'),
  // `/signals` (backend.app.engines.signal_engine) is a disconnected legacy repository that
  // nothing in the live pipeline ever writes to — it is permanently empty regardless of pipeline
  // activity. `/integration/signals` reads the real `OperationalSignal` records `_run()` actually
  // persists, which is what "Current signals" must reflect.
  signals: () => request<OperationalSignal[]>('/integration/signals?limit=20'),
  engines: () => request<EngineStatus[]>('/engines/status'),
  market: () => request<MarketStatus>('/market/status'),
  diagnostics: () => request<SystemDiagnostics>('/api/v1/system/diagnostics'),
  latestAIScore: (instrument: string, timeframe: string) => requestOptional<AIScoreSnapshot>(`/ai-scoring/latest?${scoped(instrument, timeframe)}`),
  latestSignalDecision: (instrument: string, timeframe: string) => requestOptional<SignalDecisionSnapshot>(`/signal-decisions/latest?${scoped(instrument, timeframe)}`),
  latestOperationalSignal: (instrument: string, timeframe: string) => requestOptional<OperationalSignal>(`/integration/signals/latest?${scoped(instrument, timeframe)}`),
  replays: () => request<ReplaySessionOverview[]>('/replays?limit=5'),
  pipelineStages: (instrument: string, timeframe: string) => request<PipelineStagesResponse>(`/api/v1/pipeline/stages/latest?${scoped(instrument, timeframe)}`),
  rejections: (instrument: string, timeframe: string) => request<RejectionsResponse>(`/signal-decisions/rejections/recent?${scoped(instrument, timeframe)}&limit=10`),
  marketIntelligence: (instrument: string, timeframe: string) => request<MarketIntelligence>(`/api/v1/system/market-intelligence?${scoped(instrument, timeframe)}`),
  performance: (instrument: string, timeframe: string) => request<PerformanceMetrics>(`/api/v1/system/performance?${scoped(instrument, timeframe)}`),
  aiScoreHistory: (instrument: string, timeframe: string, limit = 40) => request<AIScoreSnapshot[]>(`/ai-scoring/history?${scoped(instrument, timeframe)}&limit=${limit}`),
  chartOverlays: (instrument: string, timeframe: string, limit = 300) => request<ChartOverlays>(`/api/v1/chart/overlays?${scoped(instrument, timeframe)}&limit=${limit}`),
}

export const STREAM_URL = `${API_BASE_URL}/stream/events`
