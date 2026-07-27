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
  market_status: 'OPEN' | 'MAINTENANCE' | 'CLOSED_WEEKEND' | 'CLOSED_DAILY_BREAK' | 'HOLIDAY_OR_PROVIDER_CLOSED' | 'UNKNOWN'
  market_status_source: string
  market_timezone: string
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
  market: { symbol: string; timeframe: string; market_status: MarketStatus['market_status']; market_open: boolean; active_session: string | null; closure_reason: string | null; next_expected_open_at: string | null; server_time_utc: string; latest_candle_at: string | null; latest_candle_age_seconds: number | null; freshness: string }
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
  market_status: MarketStatus['market_status']
  market_status_source: string
  market_timezone: string
  market_closure_reason: string | null
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
    /** The canonical categorical state (see lib/economicState.ts `CalendarContextState`) — the
     * same value signal_decision_engine and the explainability layer read, so this can never
     * disagree with why a decision was or wasn't blocked. */
    context_state: string
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
      trading_context: { status: 'ready' | 'unavailable'; context_state: string; risk_window_phase: string; risk_score: number; reason: string | null }
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

export interface ProviderStatus {
  provider_name: string
  provider_version: string
  base_url: string | null
  mode: string
  enabled: boolean
  api_key_configured: boolean
  authenticated: boolean
  reachable: boolean
  stale: boolean
  rate_limited: boolean
  connection_state: string
  failure_reason: string | null
  http_status: number | null
  last_request: string | null
  last_success: string | null
  last_failure: string | null
  last_cursor: string | null
  response_time_ms: number | null
  retry_count: number
  backoff_until: string | null
  rate_limit_remaining: number | null
  rate_limit_limit: number | null
  daily_quota_used: number | null
  daily_quota_limit: number | null
  monthly_quota_used: number | null
  monthly_quota_limit: number | null
  raw_error: string | null
  message: string
  /** "keyed_api" | "public_webpage" | "rss_feed" | "ics_calendar" | "deterministic_rule" | "none".
   * For anything other than "keyed_api", the panel must never render API-key/quota fields — there
   * is no key and no quota, so showing them would actively mislead. */
  source_type: string
  robots_policy_status: string
  parser_version: string
  events_parsed: number
  last_schedule_date: string | null
  cache_age_seconds: number | null
  circuit_breaker_open: boolean
  circuit_breaker_open_until: string | null
  last_failure_category: string | null
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
 * The AI provider failed or returned something invalid; `error` then explains why. Never a fabricated
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

export interface QuantForecastHorizon {
  horizon_id: string
  timeframe: 'M1' | 'M5'
  candle_count: number
  duration_seconds: number
}

export interface QuantHorizonPrediction {
  horizon: QuantForecastHorizon
  reference_price: number
  buy_probability: number
  sell_probability: number
  neutral_probability: number
  expected_return: number
  expected_base_movement: number
  expected_minimum_movement: number
  expected_maximum_movement: number
  expected_volatility: number
  expected_mfe: number
  expected_mae: number
  tp1_probability: number
  tp2_probability: number
  stop_loss_probability: number
  sl_before_tp_probability: number
  uncertainty_interval: { low: number; high: number; confidence_level: number }
  transition_probabilities: Record<string, number>
}

export interface QuantForecastResult {
  result_id: string
  market_state_id: string
  instrument: string
  point_in_time: string
  status: 'available' | 'unavailable' | 'insufficient_history' | 'incompatible_features' | 'failed'
  model_name: string
  model_version: string
  training_dataset_version: string
  feature_schema_version: string
  calibration_version: string
  model_kind: string
  calibration_status: string
  shadow_only: true
  approved_for_publication: false
  predictions: QuantHorizonPrediction[]
  reason_codes: string[]
}

export interface QuantCalibrationReport {
  model_name: string
  model_version: string
  generated_at: string
  sample_count: number
  brier_score: number | null
  log_loss: number | null
  expected_calibration_error: number | null
  status: string
  filters: Record<string, string>
}

export interface QuantForecastOutcome {
  forecast_result_id: string
  horizon_id: string
  status: 'pending' | 'valid' | 'missing_data' | 'incomplete'
  realized_return: number | null
  realized_direction: string | null
  maximum_favorable_excursion: number | null
  maximum_adverse_excursion: number | null
  spread_adjusted_return: number | null
}

export interface AIMarketForecast {
  forecast_id: string
  status: string
  dominant_direction: 'BUY' | 'SELL' | 'NEUTRAL' | null
  buy_probability: number | null
  sell_probability: number | null
  neutral_probability: number | null
  expected_horizon: string | null
  expected_minimum_move: number | null
  expected_base_move: number | null
  expected_maximum_move: number | null
  expected_volatility: number | null
  dominant_scenario: string | null
  dominant_scenario_probability: number | null
  alternative_scenarios: { name: string; probability: number; direction: string }[]
  selected_setup_family: string | null
  supporting_evidence_ids: string[]
  contradicting_evidence_ids: string[]
  evidence_completeness: number | null
  evidence_agreement: number | null
  forecast_confidence: number | null
  uncertainty: number | null
  setup_readiness: string | null
  reasoning_summary: string | null
  failure_state: string | null
  fallback_state: string | null
  generated_at: string
  shadow_only: true
  awaiting_guardrail_validation: true
}

export interface AISignalProposal {
  proposal_id: string
  structural_opportunity_key: string
  recommended_action: string
  direction: string
  entry_type: string | null
  entry_zone: { low: number; high: number } | null
  stop_loss: number | null
  take_profit_levels: number[]
  expected_risk_to_reward: number | null
  invalidation_price: number | null
  invalidation_conditions: string[]
  expires_at: string | null
  setup_readiness: string
  proposal_confidence: number
  supporting_evidence_ids: string[]
  contradicting_evidence_ids: string[]
  shadow_only: true
  awaiting_guardrail_validation: true
}

export interface ManagedSignal {
  signal_id: string
  structural_opportunity_key: string
  setup_family: string
  direction: string
  state: string
  current_proposal_id: string
  entry_zone: { low: number; high: number } | null
  stop_loss: number | null
  take_profit_levels: number[]
  invalidation_price: number | null
  expires_at: string | null
  updated_at: string
}

export interface AIReasoningDashboard {
  forecast: AIMarketForecast | null
  proposal: AISignalProposal | null
  managed_signals: ManagedSignal[]
  signal_histories: Record<string, {
    transitions: Record<string, unknown>[]
    revisions: Record<string, unknown>[]
    monitoring: Record<string, unknown>[]
    outcomes: Record<string, unknown>[]
  }>
  final_actions: Record<string, FinalSystemAction[]>
  publications: Record<string, PublishedAnalyticalSignal | null>
  llm_usage: {
    request_count: number
    total_tokens: number | null
    successful_requests: number
    failed_requests: number
  }
  performance: Record<string, unknown> | null
  production_readiness: {
    status: string
    sample_count: number
    blockers: string[]
    warnings: string[]
    measured_checks: Record<string, { passed: boolean; threshold: unknown }>
  } | null
  runtime: {
    operating_profile: 'safe_test' | 'shadow' | 'analytical_live'
    feature_flags: Record<string, boolean>
    analytical_only: true
    broker_execution_available: false
  }
  health: {
    enabled: boolean
    proposals_enabled: boolean
    monitoring_enabled: boolean
    publication_enabled: boolean
    adjustments_enabled: boolean
    provider_available: boolean | null
    provider: string
    primary_provider: string
    active_provider: string
    fallback_status: 'ACTIVE' | 'STANDBY'
    provider_readiness: 'healthy' | 'degraded' | 'failed'
    model_identifier: string
    prompt_version: string
    latest_latency_ms: number | null
    latest_validation_passed: boolean | null
    latest_retry_count: number
    failed_requests: number
    failure_state: string | null
    fallback_state: string | null
    shadow_only: boolean
    awaiting_guardrail_validation: boolean
    providers: Record<string, {
      status: 'HEALTHY' | 'STANDBY' | 'RATE_LIMITED' | 'QUOTA_EXHAUSTED' | 'AUTH_FAILED' | 'UNAVAILABLE' | 'CIRCUIT_OPEN' | 'UNCONFIGURED'
      model: string
      last_success_at: string | null
      last_failure_at: string | null
      circuit_open_until: string | null
      last_failure_code: string | null
    }>
    deduplicated_market_states: number
    guardrails: {
      status: string
      publication_enabled: boolean
      adjustments_enabled: boolean
      analytical_only: true
      broker_execution_available: false
      actions_evaluated: number
      publications_succeeded: number
      publications_failed: number
      publication_failure_rate: number
      policy_versions: Record<string, string>
      daily_request_allowance: number
      daily_token_allowance: number
      llm_concurrency_limit: number
    }
  }
}

export type DashboardDataStatus =
  | 'available'
  | 'complete'
  | 'degraded'
  | 'partial'
  | 'pending'
  | 'failed'
  | 'not_available'
  | 'not_evaluated'
  // The backend's AI-pipeline terminal-state machine (backend/app/api/dashboard_status.py) —
  // 'pending' alone could not distinguish "genuinely fresh" from "attempted and already failed
  // during a provider-backoff window", which is exactly what let a real failure sit under the
  // same generic label indefinitely. Every one of these always carries a specific `reason`.
  | 'blocked'
  | 'disabled'
  | 'running'
  | 'wait'
  | 'not_required'
  | 'not_applicable'

export interface DashboardStage<T = unknown> {
  status: DashboardDataStatus
  reason: string
  record_id: string | null
  timestamp: string | null
  error_code: string | null
  retryable: boolean
  data: T | null
  // Present only for specific statuses — see backend/app/api/dashboard_status.py's StageResult.extra
  // for which fields accompany which status (e.g. `elapsed_seconds`/`job_state` on "running"/
  // "pending", `retry_in_seconds` on "blocked", `field`/`expected`/`received` on a structured-
  // output validation failure, `direction` on "wait", `config_source` on publication "disabled").
  elapsed_seconds?: number
  job_state?: string
  retry_in_seconds?: number
  last_failure_state?: string | null
  disabled_flags?: string[]
  field?: string
  expected?: string
  received?: unknown
  provider_http_status?: number
  upstream_reason?: string
  direction?: string
  config_source?: string
  repaired_fields?: boolean
}

export interface DashboardAggregate {
  status: 'complete' | 'partial' | 'pending' | 'failed'
  instrument: string
  generated_at: string
  correlation_id: string
  cycle: {
    event_id: string
    market_state_id: string
    analysis_timestamp: string
    knowledge_cutoff: string
    freshness: 'fresh' | 'stale'
  } | null
  stages: {
    market_state: DashboardStage
    engine_outputs: DashboardStage<unknown[]>
    quant_forecast: DashboardStage<QuantForecastResult>
    ai_reasoning: DashboardStage<AIMarketForecast>
    ai_proposal: DashboardStage<AISignalProposal>
    guardrails: DashboardStage
    final_action: DashboardStage<FinalSystemAction>
    publication: DashboardStage<PublishedAnalyticalSignal>
    monitoring: DashboardStage<ManagedSignal>
    outcome: DashboardStage
  }
  calibration: DashboardStage<QuantCalibrationReport>
  performance: DashboardStage<Record<string, unknown>>
  readiness: DashboardStage<AIReasoningDashboard['production_readiness']>
  reasoning: AIReasoningDashboard
  health: {
    quant: Record<string, unknown>
    ai: Record<string, unknown>
    guardrails: Record<string, unknown>
    feature_flags: Record<string, boolean>
  }
}

export type SystemStageStatus =
  | 'healthy'
  | 'running'
  | 'degraded'
  | 'failed'
  | 'disabled'
  | 'blocked'
  | 'stale'
  | 'no_data'

export interface SystemStageStatusItem {
  id: string
  label: string
  status: SystemStageStatus
  reason: string
  timestamp: string | null
  record_id: string | null
  details: Record<string, unknown>
}

export interface DashboardSystemStatus {
  status: 'healthy' | 'running' | 'degraded' | 'failed'
  instrument: string
  generated_at: string
  cycle_id: string | null
  stages: SystemStageStatusItem[]
  current_decision: FinalSystemAction | null
  storage: {
    status: SystemStageStatus
    reason: string
    database_bytes: number | null
    growth_bytes_per_hour: number | null
    projected_gb_per_day: number | null
    circuit_retry_at: string | null
    retention: {
      status: SystemStageStatus
      policies: Array<{
        relation_name: string
        retention_days: number
        cleanup_batch_size: number
        protected: boolean
      }>
    }
    largest_relations: Array<{
      relname: string
      total_bytes: number
      table_bytes: number
      index_bytes: number
      n_live_tup: number
      n_dead_tup: number
    }>
  }
  failure_history: Array<{
    stage: string
    status: SystemStageStatus
    reason: string
    timestamp: string | null
  }>
}

export interface FinalSystemAction {
  final_action_id: string
  action: string
  approval_state: string
  publication_state: string
  final_direction: string
  final_entry: { low: number; high: number } | null
  final_stop_loss: number | null
  final_take_profits: number[]
  final_risk_to_reward: number | null
  final_expiry: string | null
  final_risk_classification: string
  gate_evaluations: { gate_id: string; category: string; status: string; reason_codes: string[] }[]
  modifications: {
    field_name: string
    original_value: unknown
    final_value: unknown
    modifying_gate_or_policy: string
    exact_reason: string
  }[]
  policy_versions: Record<string, string>
  analytical_only: true
  broker_execution_performed: false
  created_at: string
}

export interface PublishedAnalyticalSignal {
  publication_id: string
  signal_id: string
  direction: string
  setup_family: string
  entry_zone: { low: number; high: number }
  stop_loss: number
  take_profit_levels: number[]
  lifecycle_state: string
  dominant_scenario: string
  final_risk_classification: string
  analytical_only: true
  broker_execution: false
  published_at: string
}

