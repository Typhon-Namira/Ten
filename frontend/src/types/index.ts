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
  session: string
  is_open: boolean
  checked_at: string
  note: string
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

