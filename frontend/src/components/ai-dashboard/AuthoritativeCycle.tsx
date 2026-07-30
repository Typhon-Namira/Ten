import { Activity, BrainCircuit, Clock3, History, ShieldCheck, TrendingUp } from 'lucide-react'
import type { AnalysisHistoryPage, AnalysisSignalPage, LatestCompletedCycle } from '../../types'
import { humanize } from '../../lib/aiDashboard'
import { DetailDrawer, EmptyState, Metric, SectionHeader, StatusBadge } from './Primitives'

const signalTone = (signal: string) => signal === 'BUY' ? 'positive' : signal === 'SELL' ? 'negative' : 'neutral'

function formatPrice(value: number | null | undefined) {
  return value == null ? 'Not applicable' : value.toLocaleString(undefined, { maximumFractionDigits: 3 })
}

function formatDuration(seconds: number | null | undefined) {
  if (seconds == null) return 'Not available'
  if (seconds < 60) return `${Math.round(seconds)}s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

export function CurrentAnalyticalCycle({ cycle }: { cycle: LatestCompletedCycle | null }) {
  if (!cycle || cycle.status === 'no_data' || !cycle.analytical_signal || !cycle.analysis) {
    return <section className="ai-card ai-card--wide">
      <EmptyState title="No completed analytical cycle" detail="TEN will display a signal only after analysis validation and deterministic signal persistence complete." />
    </section>
  }
  const signal = cycle.analytical_signal
  const analysis = cycle.analysis
  const decision = cycle.final_decision
  const lifecycle = cycle.signal_lifecycle
  const matrix = cycle.timeframe_matrix ?? []
  const direction = cycle.analytical_direction
  const setup = cycle.structural_trade_setup
  const execution = cycle.execution_eligibility
  const confidence = cycle.confidence_semantics
  const isCurrent = lifecycle?.remaining_validity_seconds == null || lifecycle.remaining_validity_seconds > 0
  const contribution = (name: string) => {
    const item = cycle.evidence_contributions?.[name]
    return item?.status === 'contributed' && item.weighted_contribution != null
      ? `${item.weighted_contribution.toFixed(2)} weighted (${item.evidence_count} facts)`
      : 'No qualifying contribution'
  }
  return <>
    <section className="ai-card ai-card--wide authoritative-signal">
      <SectionHeader eyebrow="Latest completed coherent cycle" title={isCurrent ? 'Current analytical signal' : 'Latest completed analytical signal'} action={<TrendingUp size={19} />} />
      {!isCurrent ? <p className="reasoning-copy"><strong>No currently valid signal.</strong> This completed signal has expired and is shown for audit only.</p> : null}
      <div className="authoritative-signal__hero">
        <StatusBadge tone={signalTone(direction?.direction ?? signal.signal)}>{direction?.direction ?? signal.signal}</StatusBadge>
        <div><strong>{signal.signal_confidence.toFixed(0)}% signal confidence · {humanize(signal.strength)}</strong><small>{signal.reasoning_summary}</small></div>
        <div className="authoritative-signal__time">
          <Clock3 size={14} />
          <span>Market time {new Date(cycle.market_time!).toLocaleString()}</span>
          <small>Cycle completed {cycle.completed_at ? new Date(cycle.completed_at).toLocaleString() : 'Not available'}</small>
          <small>Last checked {new Date(cycle.dashboard_refreshed_at).toLocaleTimeString()}</small>
        </div>
      </div>
      <div className="health-grid">
        <Metric label="Bullish score" value={direction?.bullish_score?.toFixed(2) ?? 'Not available'} />
        <Metric label="Bearish score" value={direction?.bearish_score?.toFixed(2) ?? 'Not available'} />
        <Metric label="Lifecycle" value={humanize(lifecycle?.status ?? signal.lifecycle_status)} />
        <Metric label="Signal age" value={formatDuration(lifecycle?.signal_age_seconds)} />
        <Metric label="Remaining validity" value={formatDuration(lifecycle?.remaining_validity_seconds)} />
        <Metric label="Expected horizon" value={formatDuration(signal.expected_holding_seconds)} />
      </div>
      <div className="health-grid">
        <Metric label="AI interpretation confidence" value={confidence?.ai_interpretation_confidence == null ? 'Not available' : `${confidence.ai_interpretation_confidence.toFixed(0)}%`} />
        <Metric label="Quant direction / probability" value={confidence?.quant_directional_probability == null ? 'Not available' : `${confidence.quant_direction} ${confidence.quant_directional_probability.toFixed(0)}%`} />
        <Metric label="Quant calibration" value={humanize(confidence?.quant_calibration_status ?? 'not available')} />
        <Metric label="Quant / AI direction" value={humanize(confidence?.quant_ai_alignment ?? 'unavailable')} />
        <Metric label="Evidence completeness" value={confidence?.evidence_completeness == null ? 'Not available' : `${confidence.evidence_completeness.toFixed(0)}%`} />
        <Metric label="Guardrail confidence" value={confidence == null ? 'Pending' : `${confidence.guardrail_confidence.toFixed(0)}%`} />
        <Metric label="Final overall confidence" value={confidence == null ? 'Pending' : `${confidence.final_overall_confidence.toFixed(0)}%`} />
      </div>
      <div className="health-grid">
        <Metric label="Trend contribution" value={contribution('trend')} />
        <Metric label="Institutional contribution" value={contribution('institutional')} />
        <Metric label="Volume contribution" value={contribution('volume')} />
        <Metric label="All evidence contribution" value={contribution('evidence')} />
      </div>
      {lifecycle?.outcome ? <p className="reasoning-copy">Actual RR {lifecycle.outcome.actual_risk_reward?.toFixed(2) ?? 'pending'} · MFE {formatPrice(lifecycle.outcome.maximum_favorable_excursion)} · MAE {formatPrice(lifecycle.outcome.maximum_adverse_excursion)} · Entry {lifecycle.outcome.entry_reached ? 'reached' : 'not reached'}</p> : null}
      <div className="cycle-lineage">
        <span>Cycle <code>{cycle.cycle_id}</code></span>
        <span>Analysis <code>{cycle.analysis_id}</code></span>
        <span>Signal <code>{cycle.signal_id}</code></span>
        <span>Decision <code>{cycle.decision_id ?? 'pending'}</code></span>
      </div>
    </section>
    <div className="ai-card-grid">
      <section className="ai-card">
        <SectionHeader eyebrow="Same-cycle structure" title="Structural trade setup" action={<TrendingUp size={19} />} />
        {setup == null ? <EmptyState title="No valid executable structural setup" detail="The analytical direction remains valid, but current price, reachability, structure, validity, or risk/reward did not pass the execution geometry gate." /> : <>
          <div className="health-grid">
            <Metric label="Owner timeframe" value={setup.owner_timeframe} />
            <Metric label="Direction" value={setup.direction} />
            <Metric label="Entry" value={formatPrice(setup.entry)} />
            <Metric label="Stop loss" value={formatPrice(setup.stop_loss)} />
            <Metric label="Take profit" value={formatPrice(setup.take_profit)} />
            <Metric label="Risk / reward" value={`${setup.risk_reward_ratio.toFixed(2)} / required ${setup.required_minimum_risk_reward.toFixed(2)}`} />
            <Metric label="Geometry validation" value={setup.validation_status} />
            <Metric label="Expires" value={setup.expires_at ? new Date(setup.expires_at).toLocaleString() : 'Not available'} />
          </div>
          <p className="reasoning-copy">Structural sources: {setup.basis_fact_identifiers.join(', ')}</p>
        </>}
      </section>
      <section className="ai-card">
        <SectionHeader eyebrow="Independent gating" title="Execution eligibility" action={<ShieldCheck size={19} />} />
        <StatusBadge tone={execution?.status === 'READY' ? 'positive' : 'negative'}>{execution?.status ?? 'BLOCKED'}</StatusBadge>
        <p className="reasoning-copy">{execution?.blockers.map(humanize).join(' · ') || 'No execution blockers'}</p>
      </section>
    </div>
    {matrix.length ? <section className="ai-card ai-card--wide">
      <SectionHeader eyebrow="Independent point-in-time synthesis" title="Multi-timeframe signal matrix" action={<Activity size={19} />} />
      <div className="authoritative-table">
        <div className="authoritative-table__head"><span>Timeframe</span><span>Signal</span><span>Confidence / strength</span><span>Execution</span><span>Evidence</span></div>
        {matrix.map(item => <div key={item.signal_id}>
          <strong>{item.timeframe === 'COMBINED' ? 'Combined' : item.timeframe}</strong>
          <StatusBadge tone={signalTone(item.analytical_direction)}>{item.analytical_direction}</StatusBadge>
          <span><strong>{item.confidence.toFixed(0)}%</strong><small>{humanize(item.strength)}</small></span>
          <span><strong>{item.execution_status}</strong><small>{item.blocking_reasons.map(humanize).join(' · ') || 'Eligible'}</small></span>
          <DetailDrawer label={`${item.evidence_breakdown.length} traceable facts`}>
            <p>{item.directional_thesis}</p>
            <p>Bullish {item.bullish_score.toFixed(2)} · Bearish {item.bearish_score.toFixed(2)} · Horizon {item.expected_horizon}</p>
            <p><strong>Confidence decomposition</strong> {Object.entries(item.confidence_decomposition).map(([key, value]) => `${humanize(key)} ${value.toFixed(1)}%`).join(' · ')}</p>
            <ul>{item.evidence_breakdown.map((fact, index) => <li key={`${fact.evidence_id}-${fact.correlation_group}-${index}`}>
              <strong>{fact.directional_contribution === item.analytical_direction ? 'Supporting' : 'Contradicting'} · {humanize(fact.family)} · {fact.directional_contribution} {Math.abs(fact.weighted_score).toFixed(2)}</strong>
              {' '}{fact.tool} / {fact.timeframe} · {fact.reason} · normalized {fact.normalized_score.toFixed(2)} · weight {(fact.weight * 100).toFixed(0)}% · correlation {(fact.correlated_discount * 100).toFixed(0)}% · quality {(fact.quality * 100).toFixed(0)}% · freshness {(fact.freshness * 100).toFixed(0)}% · facts {fact.source_fact_identifiers.join(', ')}
            </li>)}</ul>
            <p><strong>Invalidation</strong> {item.invalidation_conditions.join(' · ') || 'No validated structural invalidation available'}</p>
            <p><strong>Execution blockers</strong> {item.blocking_reasons.map(humanize).join(' · ') || 'None'}</p>
            <p><strong>Geometry</strong> {item.geometry == null ? 'Unavailable' : `Entry ${formatPrice(item.geometry.entry)} · stop ${formatPrice(item.geometry.stop_loss)} · target ${formatPrice(item.geometry.take_profit)} · RR ${item.geometry.risk_reward_ratio.toFixed(2)} · facts ${item.geometry.basis_fact_identifiers.join(', ')}`}</p>
          </DetailDrawer>
        </div>)}
      </div>
    </section> : null}
    <div className="ai-card-grid">
      <section className="ai-card">
        <SectionHeader eyebrow="AI interpretation" title="Deep market analysis" action={<BrainCircuit size={19} />} />
        <div className="reasoning-lead"><div><strong>{humanize(analysis.output?.market_regime.classification ?? 'not available')}</strong><small>{Math.round((analysis.output?.analysis_confidence ?? 0) * 100)}% analysis confidence</small></div></div>
        <p className="reasoning-copy">{analysis.output?.executive_summary}</p>
        <DetailDrawer label="Evidence and scenarios">
          <p>{analysis.output?.higher_timeframe_context.description}</p>
          <p>{analysis.output?.market_structure.recent_change}</p>
          <ul>{analysis.output?.alternative_scenarios.map(item => <li key={item.name}><strong>{item.name}</strong> · {Math.round(item.probability * 100)}% — {item.description}</li>)}</ul>
        </DetailDrawer>
      </section>
      <section className="ai-card">
        <SectionHeader eyebrow="Deterministic safety" title="Guardrails and publication" action={<ShieldCheck size={19} />} />
        <div className="health-grid">
          <Metric label="Final action" value={decision?.final_action ?? 'Pending'} />
          <Metric label="Readiness" value={humanize(decision?.readiness ?? 'pending')} />
          <Metric label="Publication" value={humanize(cycle.publication.status)} />
          <Metric label="Decision state" value={humanize(decision?.state ?? 'pending')} />
        </div>
        <p className="reasoning-copy">{cycle.publication.reason}</p>
        {decision?.blockers.length ? <ul>{decision.blockers.map(item => <li key={item.reason_code}>{humanize(item.reason_code)}</li>)}</ul> : null}
      </section>
    </div>
  </>
}

export function SignalHistory({ page }: { page: AnalysisSignalPage | null }) {
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Persisted deterministic outputs" title="Signal history" action={<History size={19} />} />
    {!page?.items.length ? <EmptyState title="No analytical signals in this scope" detail="Filters apply to persisted analytical signals, not legacy managed-signal records." /> :
      <div className="authoritative-table">
        <div className="authoritative-table__head"><span>Market time / cycle</span><span>Signal</span><span>Confidence / strength</span><span>Levels</span><span>Decision / outcome</span></div>
        {page.items.map(item => <div key={item.analytical_signal.signal_id}>
          <span><time>{new Date(item.analytical_signal.generated_at).toLocaleString()}</time><code>{item.analytical_signal.cycle_id}</code></span>
          <StatusBadge tone={signalTone(item.analytical_signal.signal)}>{item.analytical_signal.signal}</StatusBadge>
          <span><strong>{item.analytical_signal.confidence}%</strong><small>{humanize(item.analytical_signal.strength)}</small></span>
          <span><small>Entry {formatPrice(item.analytical_signal.entry)}</small><small>SL {formatPrice(item.analytical_signal.stop_loss)} · TP {formatPrice(item.analytical_signal.take_profit)} · RR {item.analytical_signal.risk_reward_ratio?.toFixed(2) ?? '—'}</small></span>
          <span><strong>{humanize(item.publication.status)}</strong><small>{humanize(item.guardrail_outcome)} · {humanize(item.final_action)} · {humanize(item.outcome_status)}</small></span>
        </div>)}
        <small>{page.total} persisted signals · deterministic UTC ordering</small>
      </div>}
  </section>
}

export function AnalysisHistory({ page }: { page: AnalysisHistoryPage | null }) {
  return <section className="ai-card ai-card--wide">
    <SectionHeader eyebrow="Validated provider interpretation" title="Analysis history" action={<Activity size={19} />} />
    {!page?.items.length ? <EmptyState title="No analysis history in this scope" detail="Awaiting a validated and persisted AI market analysis." /> :
      <div className="analysis-history-list">{page.items.map(({ analysis, analytical_signal }) => <details key={analysis.analysis_id}>
        <summary><time>{new Date(analysis.analysis_timestamp).toLocaleString()}</time><strong>{humanize(analysis.output?.market_regime.classification ?? analysis.status)} · {Math.round((analysis.output?.analysis_confidence ?? 0) * 100)}%</strong><StatusBadge tone={analysis.validation_passed ? 'positive' : 'negative'}>{analysis.validation_passed ? 'Schema valid' : 'Invalid'}</StatusBadge><span>{analytical_signal?.signal ?? 'No signal'}</span></summary>
        <div>
          <p>{analysis.output?.executive_summary}</p>
          <p>Cutoff {new Date(analysis.knowledge_cutoff).toLocaleString()} · HTF {humanize(analysis.output?.higher_timeframe_context.bias ?? 'not available')} · Momentum {humanize(analysis.output?.momentum_analysis.direction ?? 'not available')} · Volatility {humanize(analysis.output?.volatility_analysis.state ?? 'not available')}</p>
          <p>{analysis.provider_metadata.provider} / {analysis.provider_metadata.model} · schema {analysis.schema_version} · {analysis.provider_metadata.latency_ms == null ? 'latency not recorded' : `${analysis.provider_metadata.latency_ms.toFixed(0)} ms`}</p>
          <code>{analysis.analysis_id} · cycle {analysis.cycle_id}</code>
        </div>
      </details>)}</div>}
  </section>
}
