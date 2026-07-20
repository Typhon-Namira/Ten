export type Direction = 'long' | 'short' | 'neutral'
export type EngineState = 'ready' | 'degraded' | 'offline'

export interface Signal {
  symbol: string
  timeframe: string
  direction: Direction
  entry_zone: [number, number]
  stop_loss: number
  take_profit: number
  confidence: number
  reasoning: string[]
  risk_notes: string[]
  timestamp: string
}

export interface EngineStatus {
  name: string
  version: string
  state: EngineState
  details: string
  checked_at: string
}

export interface MarketStatus {
  symbol: string
  session: string | null
  is_open: boolean
  checked_at: string
  note: string
  market_status: 'OPEN' | 'CLOSED_WEEKEND' | 'CLOSED_DAILY_BREAK' | 'HOLIDAY_OR_PROVIDER_CLOSED' | 'UNKNOWN'
  closure_reason: string | null
  next_expected_open_at: string | null
  server_time_utc: string
  latest_candle_at: string | null
  latest_candle_age_seconds: number | null
  provider_status: string
}

export interface SystemDiagnostics {
  application_version: string
  operational_state: string
  database: { status: string; mode: string }
  provider: { name: string; configured_symbol: string; provider_symbol: string; status: string; authentication_configured: boolean; last_success_at: string | null; last_failure_at: string | null; last_error: string | null }
  market: { symbol: string; market_status: MarketStatus['market_status']; market_open: boolean; active_session: string | null; closure_reason: string | null; next_expected_open_at: string | null; server_time_utc: string; latest_candle_at: string | null; latest_candle_age_seconds: number | null; freshness: string }
  history: { candle_count: number; required_candle_count: number; initialized: boolean }
  workers: {
    market_data_worker: { enabled: boolean; running: boolean; last_heartbeat_at: string | null; last_success_at: string | null; last_error: string | null; consecutive_failures: number; processing_state: string; loaded_candles: number }
    integration_worker: { enabled: boolean; running: boolean; last_heartbeat_at: string | null; last_success_at: string | null; last_error: string | null; consecutive_failures: number }
  }
  pipeline: { status: string; latest_snapshot: Record<string, unknown> | null; latest_decision: SignalDecisionSnapshot | null; latest_scenario: OperationalSignal | null }
  replay: { enabled: boolean; status: string }
}

export type AIScoreStatus = 'ready' | 'degraded' | 'insufficient_evidence' | 'stale' | 'invalid' | 'replay'
export type DirectionalLabel = 'strong_bearish' | 'bearish' | 'slightly_bearish' | 'neutral' | 'slightly_bullish' | 'bullish' | 'strong_bullish'

export interface AIScoreSnapshot {
  snapshot_id: string
  instrument: string
  timeframe: string
  as_of: string
  policy_version: string
  directional_score: number
  directional_label: DirectionalLabel
  confidence_score: number
  market_risk_score: number
  evidence_alignment_score: number
  data_quality_score: number
  composite_score: number
  status: AIScoreStatus
  missing_sources: string[]
  degraded_sources: string[]
}

export type SignalDecisionState = 'eligible' | 'observe_only' | 'blocked' | 'insufficient_evidence' | 'expired' | 'invalid'
export type SignalDecisionDirection = 'bullish' | 'bearish' | 'neutral'

export interface DecisionReason {
  reason_code: string
  severity: string
  message_key: string
  rule_id: string
}

export interface SignalDecisionSnapshot {
  decision_id: string
  instrument: string
  timeframe: string
  direction: SignalDecisionDirection
  state: SignalDecisionState
  as_of: string
  valid_until: string
  ai_score_snapshot_id: string
  decision_policy_version: string
  eligibility_score: number
  confidence_score: number
  market_risk_score: number
  data_quality_score: number
  evidence_alignment_score: number
  blockers: DecisionReason[]
  warnings: DecisionReason[]
  mode: 'live' | 'replay'
}

export interface OperationalSignal {
  operational_signal_id: string
  semantic_hash: string
  decision_id: string
  ai_score_id: string
  snapshot_id: string
  trace_id: string
  market_event_id: string
  instrument: string
  timeframe: string
  mode: 'live' | 'replay'
  direction: string
  state: string
  confidence: number
  effective_at: string
  expires_at: string
  data_quality_status: 'valid' | 'suspect' | 'stale' | 'incomplete' | 'rejected'
  provider_provenance: string[]
  blockers: string[]
  warnings: string[]
  analytical_only: true
  trade_execution: false
}

export type ReplayStatus = 'created' | 'validating' | 'ready' | 'running' | 'pausing' | 'paused' | 'resuming' | 'cancelling' | 'cancelled' | 'completed' | 'failed' | 'recovering'
export type ReplayMode = 'maximum_speed' | 'accelerated' | 'real_time' | 'step'

export type StageStatus = 'waiting' | 'running' | 'success' | 'degraded' | 'failed' | 'skipped'

export interface PipelineStage {
  key: string
  label: string
  status: StageStatus
}

export interface PipelineStageCycle {
  symbol: string
  timeframe: string
  candle_timestamp: string
  started_at: string
  updated_at: string
  complete: boolean
  stages: PipelineStage[]
}

export interface PipelineStagesResponse extends Partial<PipelineStageCycle> {
  available: boolean
  reason?: string
}

export interface ActivityEvent {
  id: string
  type: string
  source: string
  occurred_at: string
  correlation_id: string
  payload: Record<string, unknown>
}

export type DiagnosticStatus = 'passed' | 'failed' | 'not_evaluated' | 'informational'

export interface RejectionDiagnosticEntry {
  key: string
  label: string
  status: DiagnosticStatus
  detail: string
  observed_value: unknown
  threshold: unknown
}

export interface RejectedDecision {
  decision_id: string
  instrument: string
  timeframe: string
  state: string
  direction: string
  as_of: string
  confidence_score: number
  blockers: string[]
  warnings: string[]
  ai_score_unavailable: string | null
  diagnostics: RejectionDiagnosticEntry[]
}

export interface RejectionsResponse {
  instrument: string
  timeframe: string
  count: number
  session_status_error: string | null
  rejections: RejectedDecision[]
}

export type SourceFreshness = 'fresh' | 'aging' | 'stale' | 'unknown'
export type SourceDiagnosticStatus = 'ok' | 'unavailable' | 'error'

export interface SourceDiagnostic {
  source: string
  instrument: string
  timeframe: string
  status: SourceDiagnosticStatus
  snapshot_found: boolean
  snapshot_timestamp: string | null
  age_seconds: number | null
  freshness: SourceFreshness
  error: string | null
}

export interface MarketIntelligence {
  instrument: string
  timeframe: string
  generated_at: string
  latest_candle_timestamp: string | null
  current_session: string
  market_open: boolean | null
  current_candle: { timestamp: string; open: number; high: number; low: number; close: number; volume: number; spread: number | null } | null
  spread: number | null
  current_bias: string | null
  htf_bias: Record<string, string> | null
  current_bos: { direction: string; at: string; price: number } | null
  current_choch: { direction: string; at: string; price: number } | null
  current_fvg: { type: string; upper: number; lower: number; lifecycle: string } | null
  current_order_block: { type: string; upper: number; lower: number; lifecycle: string } | null
  premium_discount: 'premium' | 'discount' | 'equilibrium' | 'unknown'
  liquidity: { available: boolean; state: Record<string, unknown> | null }
  volume_profile: { available: boolean; quality: Record<string, unknown> | null }
  institutional_flow: { available: boolean; state: Record<string, unknown> | null; quality: Record<string, unknown> | null }
  market_regime: { available: boolean; dominant_regime: string | null; trend_regime: string | null; directional_bias: string | null; trend_strength: number | null; volatility_score: number | null; confidence: number | null }
  economic_status: { available: boolean; degraded: boolean; risk_window_phase: string | null; risk_score: number | null; next_relevant_event: string | null }
  confidence_percent: number | null
  ai_directional_label: string | null
  ai_composite_score: number | null
  ai_missing_sources: string[]
  ai_degraded_sources: string[]
  scenario_readiness_percent: number | null
  decision_status: string | null
  decision_direction: string | null
  decision_active: boolean
  diagnostics: SourceDiagnostic[]
  source_errors: Record<string, string>
}

export interface PerformanceMetrics {
  instrument: string
  timeframe: string
  pipeline_latency_ms: number | null
  provider: { name: string; last_latency_ms: number | null; last_success_at: string | null; last_failure_at: string | null; last_error: string | null; healthy: boolean }
  database: { mode: string; events: number | null; outbox_backlog: number | null; processed: number | null }
  analysis: { ai_scoring: Record<string, unknown>; signal_decision: Record<string, unknown> }
  queue_length: number | null
  workers: { market_data_worker: Record<string, unknown>; integration_worker: Record<string, unknown> }
  event_bus: Record<string, unknown>
}

export interface ReplaySessionOverview {
  replay_id: string
  request_fingerprint: string
  status: ReplayStatus
  request: {
    name: string | null
    instruments: string[]
    timeframes: string[]
    start_at: string
    end_at: string
    mode: ReplayMode
    speed_multiplier: string | null
    dataset: { dataset_id: string; dataset_version: string; manifest_hash: string }
  }
  virtual_cursor_at: string
  processed_events: number
  generated_events: number
  progress_percent: string | null
  latest_checkpoint_id: string | null
  semantic_output_hash: string
  failure: { category: string; reason_code: string; detail: string } | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

