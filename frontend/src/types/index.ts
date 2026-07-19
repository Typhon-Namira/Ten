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

