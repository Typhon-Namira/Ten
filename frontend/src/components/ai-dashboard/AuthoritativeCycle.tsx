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
  return <>
    <section className="ai-card ai-card--wide authoritative-signal">
      <SectionHeader eyebrow="Latest completed coherent cycle" title="Current analytical signal" action={<TrendingUp size={19} />} />
      <div className="authoritative-signal__hero">
        <StatusBadge tone={signalTone(signal.signal)}>{signal.signal}</StatusBadge>
        <div><strong>{signal.signal_confidence.toFixed(0)}% signal confidence · {humanize(signal.strength)}</strong><small>{signal.reasoning_summary}</small></div>
        <div className="authoritative-signal__time">
          <Clock3 size={14} />
          <span>Market time {new Date(cycle.market_time!).toLocaleString()}</span>
          <small>Cycle completed {cycle.completed_at ? new Date(cycle.completed_at).toLocaleString() : 'Not available'}</small>
          <small>Last checked {new Date(cycle.dashboard_refreshed_at).toLocaleTimeString()}</small>
        </div>
      </div>
      <div className="health-grid">
        <Metric label="Entry" value={formatPrice(signal.entry)} />
        <Metric label="Stop loss" value={formatPrice(signal.stop_loss)} />
        <Metric label="Take profit" value={formatPrice(signal.take_profit)} />
        <Metric label="Risk / reward" value={signal.risk_reward_ratio?.toFixed(2) ?? 'Not applicable'} />
        <Metric label="Signal status" value={humanize(lifecycle?.status ?? signal.lifecycle_status)} />
        <Metric label="Signal age" value={formatDuration(lifecycle?.signal_age_seconds)} />
        <Metric label="Remaining validity" value={formatDuration(lifecycle?.remaining_validity_seconds)} />
        <Metric label="Expected horizon" value={formatDuration(signal.expected_holding_seconds)} />
      </div>
      <div className="health-grid">
        <Metric label="Analysis confidence" value={`${signal.analysis_confidence.toFixed(0)}%`} />
        <Metric label="Quant confidence" value={signal.quant_confidence == null ? 'Not available' : `${signal.quant_confidence.toFixed(0)}%`} />
        <Metric label="Guardrail confidence" value={decision == null ? 'Pending' : `${decision.guardrail_confidence.toFixed(0)}%`} />
        <Metric label="Overall confidence" value={`${(decision?.overall_confidence ?? signal.overall_confidence).toFixed(0)}%`} />
        <Metric label="Trend score" value={`${(signal.scoring_components.trend_alignment ?? 0).toFixed(0)}%`} />
        <Metric label="Institutional score" value={`${(signal.scoring_components.institutional_flow ?? 0).toFixed(0)}%`} />
        <Metric label="Evidence score" value={`${(signal.scoring_components.signal_quality ?? 0).toFixed(0)}%`} />
        <Metric label="Risk score" value={decision == null ? 'Pending' : `${decision.market_risk_score.toFixed(0)}%`} />
      </div>
      <p className="reasoning-copy"><strong>Quant vs AI: {humanize(signal.quant_ai_alignment)}.</strong> {signal.quant_ai_explanation}</p>
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
