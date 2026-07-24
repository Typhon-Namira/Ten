import { BrainCircuit, CandlestickChart, Sigma, Sparkles } from 'lucide-react'
import type { AIReasoningDashboard, MarketIntelligence, QuantCalibrationReport, QuantForecastResult } from '../../types'
import { humanize } from '../../lib/aiDashboard'
import { DetailDrawer, EmptyState, ProbabilityBar, SectionHeader, StatusBadge } from './Primitives'

function value(input: unknown): string {
  if (input == null || input === '') return 'Unavailable'
  return typeof input === 'string' ? humanize(input) : String(input)
}

function percent(input: number | null | undefined): string {
  return input == null ? 'Unavailable' : `${(input * 100).toFixed(0)}%`
}

export function MarketStateSummary({ data }: { data: MarketIntelligence | null }) {
  if (!data) return <section className="ai-card"><SectionHeader eyebrow="Market intelligence" title="Unified market state" /><EmptyState title="Market state unavailable" detail="Waiting for a synchronized M1, M5, and M15 state." /></section>
  const staleSources = data.diagnostics.filter(item => item.freshness === 'stale')
  return <section className="ai-card">
    <SectionHeader eyebrow="Market intelligence" title="Unified market state" action={<CandlestickChart size={19} />} />
    <div className="summary-list">
      <div><span>Market price</span><strong>{data.current_candle?.close.toFixed(2) ?? 'Unavailable'}</strong></div>
      <div><span>Regime</span><strong>{value(data.market_regime.dominant_regime)}</strong></div>
      <div><span>Structural bias</span><strong>{value(data.current_bias ?? data.market_regime.directional_bias)}</strong></div>
      <div><span>Volatility</span><strong>{percent(data.market_regime.volatility_score)}</strong></div>
      <div><span>Market status</span><strong>{value(data.market_status)}</strong></div>
      <div><span>Session</span><strong>{value(data.current_session)}</strong></div>
      <div><span>Spread</span><strong>{data.spread?.toFixed(3) ?? 'Unavailable'}</strong></div>
      <div><span>Liquidity</span><strong>{data.liquidity.available ? 'Available' : 'Unavailable'}</strong></div>
      <div><span>Event risk</span><strong>{value(data.economic_status.context_state)}</strong></div>
    </div>
    <div className="card-foot">
      <StatusBadge tone={staleSources.length ? 'negative' : 'positive'}>{staleSources.length ? `${staleSources.length} stale sources` : 'Point-in-time state current'}</StatusBadge>
      <span>M1 · M5 · M15 synchronized</span>
    </div>
    <DetailDrawer label="Technical details">
      <div className="detail-grid">
        <div><span>BOS</span><strong>{data.current_bos ? `${humanize(data.current_bos.direction)} @ ${data.current_bos.price.toFixed(2)}` : 'Unavailable'}</strong></div>
        <div><span>CHoCH</span><strong>{data.current_choch ? `${humanize(data.current_choch.direction)} @ ${data.current_choch.price.toFixed(2)}` : 'Unavailable'}</strong></div>
        <div><span>Order block</span><strong>{data.current_order_block ? humanize(data.current_order_block.lifecycle) : 'Unavailable'}</strong></div>
        <div><span>Fair value gap</span><strong>{data.current_fvg ? humanize(data.current_fvg.lifecycle) : 'Unavailable'}</strong></div>
        <div><span>Missing sources</span><strong>{data.ai_missing_sources.join(', ') || 'None'}</strong></div>
        <div><span>Degraded sources</span><strong>{data.ai_degraded_sources.join(', ') || 'None'}</strong></div>
      </div>
    </DetailDrawer>
  </section>
}

export function QuantForecastSummary({ forecast, calibration, unavailableReason }: { forecast: QuantForecastResult | null; calibration: QuantCalibrationReport | null; unavailableReason?: string }) {
  return <section className="ai-card">
    <SectionHeader eyebrow="Quantitative forecast" title="Multi-horizon outlook" action={<Sigma size={19} />} />
    {!forecast || forecast.status !== 'available' ? <EmptyState title="Forecast unavailable" detail={humanize(forecast?.reason_codes[0] ?? unavailableReason ?? 'awaiting_first_completed_cycle')} /> : <>
      <div className="horizon-stack">
        {forecast.predictions.map(item => <article className="horizon" key={item.horizon.horizon_id}>
          <div className="horizon__head"><div><strong>{item.horizon.horizon_id.replaceAll('_', ' × ')}</strong><span>{item.horizon.duration_seconds / 60} minute horizon</span></div><StatusBadge tone="neutral">{humanize(forecast.calibration_status)}</StatusBadge></div>
          <ProbabilityBar buy={item.buy_probability} sell={item.sell_probability} neutral={item.neutral_probability} />
          <div className="horizon__meta"><span>Expected move <strong>{item.expected_base_movement.toFixed(3)}</strong></span><span>Expected volatility <strong>{item.expected_volatility.toFixed(3)}</strong></span></div>
        </article>)}
      </div>
      <div className="card-foot"><span>{humanize(forecast.model_name)} · v{forecast.model_version}</span><StatusBadge tone="neutral">Shadow mode</StatusBadge></div>
      <DetailDrawer label="Model and calibration details">
        <div className="detail-grid">
          <div><span>Provider kind</span><strong>{humanize(forecast.model_kind)}</strong></div>
          <div><span>Feature schema</span><strong>{forecast.feature_schema_version}</strong></div>
          <div><span>Calibration sample</span><strong>{calibration?.sample_count ?? 0}</strong></div>
          <div><span>Brier score</span><strong>{calibration?.brier_score?.toFixed(4) ?? 'Insufficient validated sample'}</strong></div>
        </div>
      </DetailDrawer>
    </>}
  </section>
}

export function AIReasoningSummary({ data, unavailableReason }: { data: AIReasoningDashboard | null; unavailableReason?: string }) {
  const forecast = data?.forecast
  const proposal = data?.proposal
  return <section className="ai-card">
    <SectionHeader eyebrow="AI analysis" title="Market reasoning" action={<BrainCircuit size={19} />} />
    {!forecast || forecast.status === 'unavailable' || forecast.status === 'failed' || forecast.status === 'invalid' ? <EmptyState title="AI reasoning unavailable" detail={humanize(forecast?.failure_state ?? unavailableReason ?? 'awaiting_quant_forecast')} /> : <>
      <div className="reasoning-lead">
        <span className={`direction-mark direction-mark--${forecast.dominant_direction?.toLowerCase() ?? 'neutral'}`}><Sparkles size={16} />{forecast.dominant_direction ?? 'NEUTRAL'}</span>
        <div><strong>{forecast.dominant_scenario ?? 'Scenario unavailable'}</strong><small>{forecast.generated_at ? new Date(forecast.generated_at).toLocaleString() : 'Timestamp unavailable'}</small></div>
      </div>
      <div className="summary-list">
        <div><span>Setup family</span><strong>{value(forecast.selected_setup_family)}</strong></div>
        <div><span>Forecast confidence</span><strong>{percent(forecast.forecast_confidence)}</strong></div>
        <div><span>Proposal</span><strong>{value(proposal?.recommended_action)}</strong></div>
        <div><span>Readiness</span><strong>{value(proposal?.setup_readiness)}</strong></div>
        <div><span>Invalidation</span><strong>{proposal?.invalidation_conditions[0] ?? 'Unavailable'}</strong></div>
        <div><span>Key risk</span><strong>{proposal?.contradicting_evidence_ids.length ? `${proposal.contradicting_evidence_ids.length} contradicting evidence items` : 'No contradiction identified'}</strong></div>
      </div>
      <div className="card-foot"><span>{data?.health.provider} · {data?.health.model_identifier}</span><StatusBadge tone={proposal ? 'positive' : 'neutral'}>{proposal ? 'Validated proposal' : 'No proposal'}</StatusBadge></div>
      <DetailDrawer label="Full reasoning and lineage">
        <p className="reasoning-copy">{forecast.reasoning_summary ?? 'No controlled reasoning summary was returned.'}</p>
        <div className="detail-grid">
          <div><span>Alternative scenarios</span><strong>{forecast.alternative_scenarios.map(item => `${item.name} ${(item.probability * 100).toFixed(0)}%`).join(', ') || 'None'}</strong></div>
          <div><span>Supporting evidence</span><strong>{forecast.supporting_evidence_ids.length}</strong></div>
          <div><span>Contradicting evidence</span><strong>{forecast.contradicting_evidence_ids.length}</strong></div>
          <div><span>Prompt version</span><strong>{data?.health.prompt_version ?? 'Unavailable'}</strong></div>
        </div>
      </DetailDrawer>
    </>}
  </section>
}
