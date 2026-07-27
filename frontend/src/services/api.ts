import type { ActiveSelection, AIReasoningDashboard, AIMarketAnalysis, AIScoreSnapshot, ChartOverlays, ChatTurn, DashboardAggregate, DashboardSystemStatus, EngineStatus, ExplainResponse, MarketIntelligence, MarketStatus, OperationalSignal, PerformanceMetrics, PipelineStagesResponse, QuantCalibrationReport, QuantForecastOutcome, QuantForecastResult, RejectionsResponse, ReplaySessionOverview, SignalDecisionSnapshot, SystemDiagnostics } from '../types'
import { ApiError } from '../lib/apiError'

const API_BASE_URL = import.meta.env.VITE_API_URL?.replace(/\/$/, '') ?? ''

// `/api/v1/dashboard/latest` aggregates every analytical engine's evidence into one response and
// has been observed in the multi-megabyte range on a live deployment — generous but bounded, so a
// genuinely hung backend still surfaces as a distinguishable timeout instead of an indefinite spinner.
const DEFAULT_TIMEOUT_MS = 20_000
const LARGE_PAYLOAD_TIMEOUT_MS = 45_000

async function timedFetch(path: string, init: RequestInit = {}, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<Response> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('timeout', `Request to ${path} did not complete within ${timeoutMs / 1000}s`)
    }
    throw new ApiError('network', error instanceof Error ? error.message : 'Network request failed')
  } finally {
    window.clearTimeout(timer)
  }
}

async function errorDetail(response: Response): Promise<string> {
  let detail = response.statusText || 'request_failed'
  try {
    const body = await response.json() as { detail?: string; reason?: string }
    detail = body.reason ?? body.detail ?? detail
  } catch {
    // The status code and correlation-aware backend logs remain authoritative.
  }
  return detail
}

async function parseJson<T>(response: Response, path: string): Promise<T> {
  try {
    return (await response.json()) as T
  } catch (error) {
    throw new ApiError('parse', `${path} returned a body that could not be parsed as JSON: ${error instanceof Error ? error.message : String(error)}`)
  }
}

async function request<T>(path: string, timeoutMs?: number): Promise<T> {
  const response = await timedFetch(path, {}, timeoutMs)
  if (!response.ok) throw new ApiError('http', await errorDetail(response), response.status)
  return parseJson<T>(response, path)
}

async function requestOptional<T>(path: string): Promise<T | null> {
  const response = await timedFetch(path)
  if (response.status === 404) return null
  if (!response.ok) throw new ApiError('http', await errorDetail(response), response.status)
  return parseJson<T>(response, path)
}

async function requestJson<T>(path: string, body: unknown): Promise<T> {
  const response = await timedFetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }, LARGE_PAYLOAD_TIMEOUT_MS)
  if (!response.ok) throw new ApiError('http', await errorDetail(response), response.status)
  return parseJson<T>(response, path)
}

function isDashboardAggregate(value: unknown): value is DashboardAggregate {
  if (value == null || typeof value !== 'object') return false
  const candidate = value as Partial<DashboardAggregate>
  const stages = candidate.stages
  return (
    typeof candidate.status === 'string'
    && typeof candidate.instrument === 'string'
    && stages != null
    && typeof stages === 'object'
    && stages.market_state != null
    && stages.quant_forecast != null
    && stages.ai_reasoning != null
    && stages.final_action != null
    && candidate.reasoning != null
  )
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
    const response = await timedFetch(path)
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
  latestQuantForecast: (instrument: string) => requestOptional<QuantForecastResult>(`/api/v1/quant-forecasts/latest?instrument=${encodeURIComponent(instrument)}`),
  latestQuantCalibration: () => requestOptional<QuantCalibrationReport>('/api/v1/quant-forecasts/calibration/latest'),
  quantForecastOutcomes: (resultId: string) => request<QuantForecastOutcome[]>(`/api/v1/quant-forecasts/${encodeURIComponent(resultId)}/outcomes`),
  latestAIReasoning: (instrument: string) => request<AIReasoningDashboard>(`/api/v1/ai-reasoning/latest?instrument=${encodeURIComponent(instrument)}`),
  aiAnalysisHistory: (instrument: string, timeframe: string, cursor = 0, limit = 100) => request<AIMarketAnalysis[]>(`/api/v1/ai-reasoning/analyses?instrument=${encodeURIComponent(instrument)}&timeframe=${encodeURIComponent(timeframe)}&cursor=${cursor}&limit=${limit}`),
  dashboardLatest: async (instrument: string) => {
    const value = await request<unknown>(`/api/v1/dashboard/latest?instrument=${encodeURIComponent(instrument)}`, LARGE_PAYLOAD_TIMEOUT_MS)
    if (!isDashboardAggregate(value)) {
      // A genuinely malformed/renamed response shape — NOT the same thing as `status: "failed"`
      // or any individual `stages.*.status === "failed"` inside an otherwise well-shaped payload;
      // those are legitimate backend-reported outcomes and pass this check fine (see
      // isDashboardAggregate above, which only checks presence/type, never semantic status values).
      throw new ApiError('parse', 'TEN dashboard response schema mismatch — missing required stages/reasoning fields; the backend contract may have changed')
    }
    return value
  },
  dashboardSystemStatus: (instrument: string) =>
    request<DashboardSystemStatus>(`/api/dashboard/system-status?instrument=${encodeURIComponent(instrument)}`),
  // Explainability: every AI-authored answer here is prose over a context TEN itself assembled —
  // the AI never fetches its own data, so a chat answer can never disagree with these same panels.
  explainCurrent: (instrument: string, timeframe: string) => request<ExplainResponse>(`/api/v1/explain/current?${scoped(instrument, timeframe)}`),
  explainDecision: (decisionId: string) => request<ExplainResponse>(`/api/v1/explain/decision/${decisionId}`),
  explainRejection: (decisionId: string) => request<ExplainResponse>(`/api/v1/explain/rejection/${decisionId}`),
  explainChat: (message: string, history: ChatTurn[], instrument: string, timeframe: string) => requestJson<ExplainResponse>('/api/v1/explain/chat', { message, history, instrument, timeframe }),
}

export const STREAM_URL = `${API_BASE_URL}/stream/events`
