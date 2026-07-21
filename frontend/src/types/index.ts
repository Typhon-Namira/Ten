export type EngineState = 'ready' | 'degraded' | 'offline'

/** The single authoritative (instrument, timeframe) pair the pipeline actually runs — every
 * dashboard data source must be queried with this pair, never a hardcoded literal, so widgets
 * can't silently disagree about which candle series they're each showing. */
export interface ActiveSelection {
  instrument: string
  timeframe: string
  configured_instruments: string[]
  configured_timeframes: string[]
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
  /** >1 when the backend merged a burst of same-type, same-cycle events into one entry (e.g. N
   * liquidity pools swept in one analysis pass) instead of emitting N near-duplicate log lines. */
  count: number
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
  economic_status: {
    available: boolean
    degraded: boolean
    risk_window_phase: string | null
    risk_score: number | null
    next_relevant_event: string | null
    /** Provider health -> downloaded events -> mapped events -> relevant events -> trading
     * context, each independently reported — null only if the calendar sync itself failed. */
    stages: {
      provider_health: { status: 'healthy' | 'unavailable'; reachable_providers: string[] }
      downloaded_events: { status: 'ok' | 'empty'; count: number }
      mapped_events: { status: 'ok' | 'degraded'; mapped_count: number; unmapped_count: number }
      relevant_events: { status: 'available' | 'none_relevant'; active_count: number; has_previous_event: boolean; has_next_event: boolean }
      trading_context: { status: 'ready' | 'unavailable'; risk_window_phase: string; risk_score: number; reason: string | null }
    } | null
  }
  confidence_percent: number | null
  risk_percent: number | null
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
  /** Elapsed time of the currently-running cycle, if one is in flight (not yet complete). */
  pipeline_in_flight_ms: number | null
  /** Age of the oldest queued-but-unprocessed outbox item; populated whenever `queue_length > 0`
   * even if `pipeline_latency_ms`/`pipeline_in_flight_ms` are both null. */
  queue_oldest_pending_age_seconds: number | null
  provider: {
    name: string
    last_latency_ms: number | null
    last_success_at: string | null
    last_failure_at: string | null
    last_error: string | null
    healthy: boolean
    consecutive_failures: number | null
    provider_backoff_until: string | null
    provider_rate_limit_remaining: number | null
    provider_rate_limit: number | null
  }
  database: { mode: string; events: number | null; outbox_backlog: number | null; processed: number | null; last_database_update: string | null; latest_market_candle: string | null }
  cache: { last_cache_update: string | null; hit_ratio: number; writes: number }
  analysis: { ai_scoring: Record<string, unknown>; signal_decision: Record<string, unknown> }
  queue_length: number | null
  workers: { market_data_worker: Record<string, unknown>; integration_worker: Record<string, unknown> }
  event_bus: Record<string, unknown>
}

export interface ChartCandle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface ChartStructureEvent {
  id: string
  kind: string
  direction: string
  time: number
  price: number
  confidence: number
}

export interface ChartZone {
  id: string
  kind: string
  direction: string
  upper: number
  lower: number
  start_time: number
  lifecycle_state: string
  mitigation_percentage: number
}

export interface ChartDealingRange {
  range_high: number
  range_low: number
  equilibrium: number
  premium_boundary: number
  discount_boundary: number
  golden_zone_low: number
  golden_zone_high: number
  start_time: number
  end_time: number
  direction: string
}

export interface ChartLiquidityPool {
  id: string
  side: string
  upper: number
  lower: number
  start_time: number
  lifecycle_state: string
  strength: number
  target_rank: number | null
}

export interface ChartLiquiditySweep {
  id: string
  kind: string
  time: number
  price: number
  side: string
}

export interface ChartSession {
  session: string
  high: number
  low: number
  opened_at: number
  completed: boolean
}

export interface ChartVolumeProfile {
  poc: number | null
  vah: number | null
  val: number | null
  start_time: number
  end_time: number
}

export interface ChartEqualLevel {
  id: string
  side: string
  price: number
  time: number
  member_count: number
}

export interface ChartEconomicEvent {
  id: string
  name: string
  importance: string
  time: number
}

export interface ChartDecisionAnnotation {
  direction: string
  state: string
  confidence: number
  time: number
}

export interface ChartOverlays {
  instrument: string
  timeframe: string
  generated_at: string
  candles: ChartCandle[]
  structure_events: ChartStructureEvent[]
  zones: ChartZone[]
  dealing_range: ChartDealingRange | null
  liquidity_pools: ChartLiquidityPool[]
  liquidity_sweeps: ChartLiquiditySweep[]
  equal_levels: ChartEqualLevel[]
  sessions: ChartSession[]
  volume_profile: ChartVolumeProfile | null
  economic_events: ChartEconomicEvent[]
  decision: ChartDecisionAnnotation | null
  source_errors: Record<string, string>
}

export interface EngineInfluence {
  engine: string
  influence: string
  note: string
}

export interface Explanation {
  summary: string
  primary_reasons: string[]
  opposing_factors: string[]
  engine_breakdown: EngineInfluence[]
  required_for_change: string[]
  caveats: string[]
}

export interface Evidence {
  source: string
  reference_id: string
  timestamp: string | null
}

export interface EngineFact {
  engine: string
  available: boolean
  summary: Record<string, unknown>
  evidence: Evidence | null
  error: string | null
}

export interface ExplainabilityScore {
  percent: number
  engines_available: number
  engines_total: number
  evidence_citations: number
  has_ai_score: boolean
  has_decision: boolean
}

/** Shared shape for every `/api/v1/explain/*` response — `explanation` is `null` only when
 * OpenRouter failed or returned something invalid; `error` then explains why. Never a fabricated
 * explanation standing in for a real one. */
export interface ExplainResponse {
  instrument: string
  timeframe: string
  generated_at: string
  explanation: Explanation | null
  error: string | null
  explainability_score: ExplainabilityScore
  evidence: Evidence[]
  engines?: EngineFact[]
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
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

