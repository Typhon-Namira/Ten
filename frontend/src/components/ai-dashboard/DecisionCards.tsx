import { Activity, Gauge, ShieldCheck, Signal, TimerReset } from 'lucide-react'
import type { AIReasoningDashboard, FinalSystemAction, MarketIntelligence } from '../../types'
import { activePublication, activeSignal, humanize, latestFinalAction, numberValue } from '../../lib/aiDashboard'
import { DetailDrawer, EmptyState, Metric, SectionHeader, StatusBadge } from './Primitives'

function gateCounts(action: FinalSystemAction | null) {
  const evaluations = action?.gate_evaluations ?? []
  return {
    passed: evaluations.filter(item => item.status === 'passed').length,
    failed: evaluations.filter(item => item.status === 'failed').length,
    unavailable: evaluations.filter(item => item.status === 'unavailable').length,
  }
}

export function GuardrailSummary({ data, unavailableReason }: { data: AIReasoningDashboard | null; unavailableReason?: string }) {
  const action = latestFinalAction(data)
  const counts = gateCounts(action)
  const blocker = action?.gate_evaluations.find(item => item.status === 'failed' || item.status === 'unavailable')
  const categories = ['technical', 'execution', 'risk']
  return <section className="ai-card">
    <SectionHeader eyebrow="Deterministic safety" title="Guardrail decision" action={<ShieldCheck size={19} />} />
    {!action ? <EmptyState title="Guardrails not evaluated" detail={humanize(unavailableReason ?? 'awaiting_ai_proposal')} /> : <>
      <div className="guardrail-result">
        <div className={`guardrail-result__mark guardrail-result__mark--${counts.failed ? 'negative' : 'positive'}`}><ShieldCheck size={23} /></div>
        <div><strong>{humanize(action.approval_state)}</strong><span>{blocker ? `${humanize(blocker.gate_id)} · ${humanize(blocker.reason_codes[0])}` : 'All required gates passed'}</span></div>
      </div>
      <div className="guardrail-counts">
        <Metric label="Passed" value={counts.passed} />
        <Metric label="Failed" value={counts.failed} />
        <Metric label="Unavailable" value={counts.unavailable} />
      </div>
      <div className="card-foot"><span>Evaluated {new Date(action.created_at).toLocaleString()}</span><span>{action.policy_versions.hard_gates ?? action.policy_versions.guardrails}</span></div>
      <DetailDrawer label="Review all gate categories">
        <div className="gate-groups">
          {categories.map(category => <div key={category}><h4>{humanize(category)} gates</h4>
            {action.gate_evaluations.filter(gate => gate.category === category).map(gate => <div className="gate-row" key={gate.gate_id}>
              <StatusBadge tone={gate.status === 'passed' ? 'positive' : gate.status === 'failed' ? 'negative' : 'neutral'}>{humanize(gate.status)}</StatusBadge>
              <span><strong>{humanize(gate.gate_id)}</strong><small>{humanize(gate.reason_codes[0] ?? 'No blocking reason')}</small></span>
            </div>)}
          </div>)}
        </div>
      </DetailDrawer>
    </>}
  </section>
}

function recordValue(record: Record<string, unknown> | undefined, key: string): string {
  const value = record?.[key]
  if (typeof value === 'number') return value.toFixed(3)
  if (typeof value === 'string') return humanize(value)
  return 'Unavailable'
}

export function ActiveSignalMonitor({ data, market, unavailableReason }: { data: AIReasoningDashboard | null; market: MarketIntelligence | null; unavailableReason?: string }) {
  const signal = activeSignal(data)
  const publication = activePublication(data)
  if (!signal || !publication) return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Persistent monitoring" title="Active analytical signal" action={<Signal size={19} />} />
    <EmptyState title="No active analytical signal" detail={humanize(unavailableReason ?? 'no_managed_signal_for_latest_cycle')} />
  </section>
  const history = data?.signal_histories[signal.signal_id]
  const revision = history?.revisions.at(-1)
  const outcome = history?.outcomes.at(-1)
  const midpoint = (publication.entry_zone.low + publication.entry_zone.high) / 2
  const current = market?.current_candle?.close
  const movement = current == null ? null : publication.direction === 'BUY' ? current - midpoint : midpoint - current
  const lifecycle = ['proposed', 'confirmed', 'waiting_for_entry', 'active', 'partially_realized', 'tp1_hit', 'tp2_hit', 'closed']
  const currentIndex = lifecycle.indexOf(signal.state)
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Persistent monitoring" title="Active analytical signal" action={<StatusBadge tone={publication.direction === 'BUY' ? 'positive' : 'negative'}>{publication.direction}</StatusBadge>} />
    <div className="signal-monitor">
      <div className="signal-monitor__primary">
        <strong>{signal.setup_family.replaceAll('_', ' ')}</strong>
        <span>Published {new Date(publication.published_at).toLocaleString()}</span>
        <div className="signal-monitor__levels">
          <Metric label="Entry" value={`${publication.entry_zone.low.toFixed(2)} – ${publication.entry_zone.high.toFixed(2)}`} />
          <Metric label="Current" value={current?.toFixed(2) ?? 'Unavailable'} />
          <Metric label="Movement" value={movement == null ? 'Unavailable' : `${movement >= 0 ? '+' : ''}${movement.toFixed(2)}`} />
          <Metric label="Stop loss" value={publication.stop_loss.toFixed(2)} />
          <Metric label="Targets" value={publication.take_profit_levels.map(item => item.toFixed(2)).join(' · ')} />
          <Metric label="Outcome" value={recordValue(outcome, 'status')} />
        </div>
      </div>
      <div className="signal-lifecycle" aria-label={`Signal lifecycle: ${humanize(signal.state)}`}>
        {lifecycle.map((state, index) => <div className={index < currentIndex ? 'done' : index === currentIndex ? 'current' : ''} key={state}><i /><span>{humanize(state)}</span></div>)}
      </div>
    </div>
    <div className="card-foot"><span>Last monitoring update {new Date(signal.updated_at).toLocaleString()}</span><span>{revision ? `Latest revision: ${recordValue(revision, 'level_type')}` : 'No approved revisions'}</span></div>
    <DetailDrawer label="Monitoring and revision history">
      <div className="detail-grid">
        <div><span>Monitoring evaluations</span><strong>{history?.monitoring.length ?? 0}</strong></div>
        <div><span>Level revisions</span><strong>{history?.revisions.length ?? 0}</strong></div>
        <div><span>Transitions</span><strong>{history?.transitions.length ?? 0}</strong></div>
        <div><span>Stop protection</span><strong>Widening prohibited</strong></div>
        <div><span>MFE</span><strong>{recordValue(outcome, 'maximum_favorable_excursion')}</strong></div>
        <div><span>MAE</span><strong>{recordValue(outcome, 'maximum_adverse_excursion')}</strong></div>
      </div>
    </DetailDrawer>
  </section>
}

function performanceComparison(data: AIReasoningDashboard | null): Record<string, Record<string, unknown>> {
  const report = data?.performance
  if (!report || typeof report.comparison !== 'object' || report.comparison == null) return {}
  return report.comparison as Record<string, Record<string, unknown>>
}

export function ValidationSummary({ data, unavailableReason }: { data: AIReasoningDashboard | null; unavailableReason?: string }) {
  const readiness = data?.production_readiness
  const comparison = performanceComparison(data)
  const approved = comparison.guardrail_approved ?? {}
  const sample = readiness?.sample_count ?? numberValue(data?.performance ?? null, 'sample_count') ?? 0
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Validation" title="Performance and production readiness" action={<Gauge size={19} />} />
    <div className="validation-grid">
      <div className="validation-metrics">
        <Metric label="Evaluated signals" value={sample || 'Insufficient validated sample'} />
        <Metric label="Expected value" value={numberValue(approved, 'expected_value')?.toFixed(3) ?? 'Insufficient validated sample'} />
        <Metric label="Average realized R:R" value={numberValue(approved, 'average_risk_to_reward')?.toFixed(2) ?? 'Insufficient validated sample'} />
        <Metric label="Maximum drawdown" value={numberValue(approved, 'maximum_drawdown')?.toFixed(3) ?? 'Insufficient validated sample'} />
      </div>
      <div className="readiness">
        <div className="readiness__head"><StatusBadge tone={readiness?.status === 'ready_for_analytical_live' ? 'positive' : 'negative'}>{humanize(readiness?.status ?? 'not evaluated')}</StatusBadge><span>{readiness ? `${Object.values(readiness.measured_checks).filter(check => check.passed).length}/${Object.keys(readiness.measured_checks).length} checks passed` : 'No readiness report'}</span></div>
        <p>{readiness?.status === 'ready_for_analytical_live' ? 'Backend validation has approved analytical-live eligibility.' : 'Publication remains ineligible until backend readiness blockers and sample requirements pass.'}</p>
        <ul>{readiness?.blockers.slice(0, 4).map(blocker => <li key={blocker}><TimerReset size={14} />{humanize(blocker)}</li>) ?? <li><TimerReset size={14} />{humanize(unavailableReason ?? 'insufficient_validated_sample')}</li>}</ul>
      </div>
    </div>
    <DetailDrawer label="Compare validation systems">
      <div className="comparison-grid">
        {['legacy', 'quantitative_shadow', 'ai_proposals', 'guardrail_approved'].map(name => {
          const metrics = comparison[name] ?? {}
          return <div key={name}><strong>{humanize(name)}</strong><span>Sample {numberValue(metrics, 'completed_sample_size') ?? 0}</span><span>Expected value {numberValue(metrics, 'expected_value')?.toFixed(3) ?? 'Unavailable'}</span><span>Profit factor {numberValue(metrics, 'profit_factor')?.toFixed(2) ?? 'Unavailable'}</span></div>
        })}
      </div>
    </DetailDrawer>
  </section>
}

export function OperationalHealth({ data, stale }: { data: AIReasoningDashboard | null; stale: boolean }) {
  const health = data?.health
  const usage = data?.llm_usage
  const accounts = ['groq_1', 'groq_2', 'groq_3', 'groq_4'] as const
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Operations" title="AI usage and system health" action={<Activity size={19} />} />
    <div className="health-grid">
      <Metric label="Five-minute policy calls" value={usage?.five_minute_policy?.provider_http_calls ?? 'Unavailable'} detail={`Allowance ${health?.guardrails.daily_request_allowance ?? 'Unavailable'}`} />
      <Metric label="Legacy daily calls" value={usage?.legacy_cumulative_daily?.provider_http_calls ?? 'Unavailable'} detail="Pre-policy telemetry retained" />
      <Metric label="Eligible 5-minute cycles" value={health?.call_control.eligible_five_minute_cycles ?? 'Unavailable'} />
      <Metric label="Analyses completed" value={health?.call_control.analyses_successfully_completed ?? 'Unavailable'} />
      <Metric label="Total Groq calls" value={usage?.groq_calls ?? 'Unavailable'} />
      <Metric label="Configured accounts" value={health?.configured_account_count ?? 0} />
      <Metric label="Available accounts" value={health?.available_account_count ?? 0} />
      <Metric label="Retries" value={usage?.retries ?? 'Unavailable'} />
      <Metric label="Schema corrections" value={usage?.schema_corrections ?? 'Unavailable'} />
      <Metric label="Initial parse failures" value={usage?.initial_parse_failures ?? 'Unavailable'} />
      <Metric label="Initial schema failures" value={usage?.initial_schema_validation_failures ?? 'Unavailable'} />
      <Metric label="Corrections succeeded / failed" value={`${usage?.schema_corrections_succeeded ?? 0} / ${usage?.schema_corrections_failed ?? 0}`} />
      <Metric label="HTTP 429 responses" value={usage?.http_429_responses ?? 'Unavailable'} />
      <Metric label="Truncated outputs" value={usage?.truncated_outputs ?? 'Unavailable'} detail={usage?.truncation_rate == null ? 'Rate unavailable' : `${(usage.truncation_rate * 100).toFixed(1)}%`} />
      <Metric label="Compact retries" value={usage?.compact_retries ?? 'Unavailable'} />
      <Metric label="Request-policy failures" value={usage?.request_policy_failures ?? 'Unavailable'} />
      <Metric label="Tokens / completed analysis" value={usage?.tokens_per_completed_analysis?.toLocaleString() ?? 'Unavailable'} />
      <Metric label="Calls / completed analysis" value={usage?.provider_calls_per_completed_analysis?.toFixed(2) ?? 'Unavailable'} />
      <Metric label="Average input tokens" value={usage?.average_input_tokens?.toFixed(0) ?? 'Unavailable'} />
      <Metric label="Average output tokens" value={usage?.average_output_tokens?.toFixed(0) ?? 'Unavailable'} detail={`P95 ${usage?.p95_output_tokens ?? 'Unavailable'}`} />
      <Metric label="Completion rate" value={usage?.completion_rate == null ? 'Unavailable' : `${(usage.completion_rate * 100).toFixed(1)}%`} />
      <Metric label="Skipped before provider" value={health?.call_control.skipped_before_provider_call ?? 'Unavailable'} />
      <Metric label="Deduplicated before provider" value={health?.call_control.deduplicated_before_provider_call ?? 'Unavailable'} />
      <Metric label="Provider failures" value={usage?.provider_failures ?? 'Unavailable'} />
      <Metric label="Validation failures" value={usage?.validation_failures ?? 'Unavailable'} />
      <Metric label="Recorded tokens" value={usage?.total_tokens?.toLocaleString() ?? 'Unavailable'} detail={`Allowance ${health?.guardrails.daily_token_allowance?.toLocaleString() ?? 'Unavailable'}`} />
      <Metric label="Failures" value={usage?.failed_requests ?? health?.failed_requests ?? 'Unavailable'} />
      <Metric label="Latest retry count" value={health?.latest_retry_count ?? 'Unavailable'} />
      <Metric label="Concurrency limit" value={health?.guardrails.llm_concurrency_limit ?? 'Unavailable'} />
      <Metric label="AI latency" value={health?.latest_latency_ms == null ? 'Unavailable' : `${health.latest_latency_ms.toFixed(0)} ms`} />
      <Metric label="Data freshness" value={stale ? 'Stale' : 'Fresh'} />
      {accounts.map((accountId, index) => {
        const account = health?.providers[accountId]
        return <Metric
          key={accountId}
          label={`Groq ${index + 1}`}
          value={account?.status ?? 'DISABLED'}
          detail={`${account?.eligible_now ? 'eligible' : 'not eligible'} · ${account?.circuit_state ?? 'CLOSED'} · HTTP ${account?.last_http_status ?? 'none'} · result ${account?.last_request_result ?? 'none'} · policy ${account?.request_policy_health ?? 'healthy'} · analysis requests ${account?.analysis_requests ?? 0} · corrections ${account?.schema_correction_requests ?? 0} · 429s ${account?.http_429_responses ?? 0} · tokens ${account?.token_usage.total_tokens ?? 0} · persisted analyses ${account?.successful_analyses ?? 0} · reset ${account?.cooldown_until ?? 'none'}`}
        />
      })}
    </div>
    {(usage?.recent_provider_attempts?.length ?? 0) > 0 && <div className="comparison-grid">
      {usage!.recent_provider_attempts.slice(0, 8).map(attempt => <div key={attempt.provider_attempt_id}>
        <strong>{attempt.account_id} · {humanize(attempt.request_kind)}</strong>
        <span>HTTP {attempt.http_status ?? 'none'} · finish {attempt.finish_reason ?? 'unknown'}</span>
        <span>Tokens {attempt.input_tokens ?? '—'} / {attempt.output_tokens ?? '—'} / {attempt.total_tokens ?? '—'}</span>
        <span>{attempt.output_profile ?? 'unknown profile'} · budget {attempt.target_output_tokens ?? '—'} / {attempt.hard_output_limit ?? '—'} · used {attempt.output_budget_utilization_percent ?? '—'}%</span>
        <span>Schema {attempt.schema_valid === true ? 'valid' : attempt.schema_error_code ?? 'not validated'} {attempt.schema_error_path ?? ''}</span>
      </div>)}
    </div>}
    <div className="card-foot">
      <span>Status {health?.operations_status ?? 'idle'} · primary {health?.primary_provider ?? 'Groq pool'} · active {health?.active_provider ?? 'none'} · {health?.model_identifier ?? 'Model unavailable'}</span>
      <StatusBadge tone={health?.operations_status === 'healthy' && !stale ? 'positive' : 'warning'}>{humanize(health?.operations_status ?? 'idle')}</StatusBadge>
    </div>
  </section>
}
