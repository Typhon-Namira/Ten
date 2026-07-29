import { BrainCircuit, CandlestickChart, Sigma, Sparkles } from 'lucide-react'
import type { AIReasoningDashboard, MarketIntelligence, QuantCalibrationReport, QuantForecastResult } from '../../types'
import { humanize } from '../../lib/aiDashboard'
import { DetailDrawer, EmptyState, ProbabilityBar, SectionHeader, StatusBadge } from './Primitives'

function percent(input: number | null | undefined): string {
  return input == null ? '' : `${(input * 100).toFixed(0)}%`
}

export function MarketStateSummary({ data }: { data: MarketIntelligence | null }) {
  if (!data) return <section className="ai-card"><SectionHeader eyebrow="Market intelligence" title="Unified market state" /><EmptyState title="Waiting for market state" detail="Waiting for a synchronized M5 and M15 state." /></section>
  const staleSources = data.diagnostics.filter(item => item.freshness === 'stale')
  return <section className="ai-card">
    <SectionHeader eyebrow="Market intelligence" title="Unified market state" action={<CandlestickChart size={19} />} />
    <div className="summary-list">
      {data.current_candle && <div><span>Market price</span><strong>{data.current_candle.close.toFixed(2)}</strong></div>}
      {data.market_regime.dominant_regime && <div><span>Regime</span><strong>{humanize(data.market_regime.dominant_regime)}</strong></div>}
      {(data.current_bias ?? data.market_regime.directional_bias) && <div><span>Structural bias</span><strong>{humanize(data.current_bias ?? data.market_regime.directional_bias ?? '')}</strong></div>}
      {data.market_regime.volatility_score != null && <div><span>Volatility</span><strong>{percent(data.market_regime.volatility_score)}</strong></div>}
      {data.market_status && <div><span>Market status</span><strong>{humanize(data.market_status)}</strong></div>}
      {data.current_session && <div><span>Session</span><strong>{humanize(data.current_session)}</strong></div>}
      {data.spread != null && <div><span>Spread</span><strong>{data.spread.toFixed(3)}</strong></div>}
      {data.liquidity.available && <div><span>Liquidity</span><strong>Available</strong></div>}
      {data.economic_status.context_state && <div><span>Event risk</span><strong>{humanize(data.economic_status.context_state)}</strong></div>}
    </div>
    <div className="card-foot">
      <StatusBadge tone={staleSources.length ? 'negative' : 'positive'}>{staleSources.length ? `${staleSources.length} stale sources` : 'Point-in-time state current'}</StatusBadge>
      <span>M5 · M15 synchronized</span>
    </div>
    {(data.current_bos || data.current_choch || data.current_order_block || data.current_fvg || data.ai_missing_sources.length || data.ai_degraded_sources.length) && <DetailDrawer label="Technical details">
      <div className="detail-grid">
        {data.current_bos && <div><span>BOS</span><strong>{humanize(data.current_bos.direction)} @ {data.current_bos.price.toFixed(2)}</strong></div>}
        {data.current_choch && <div><span>CHoCH</span><strong>{humanize(data.current_choch.direction)} @ {data.current_choch.price.toFixed(2)}</strong></div>}
        {data.current_order_block && <div><span>Order block</span><strong>{humanize(data.current_order_block.lifecycle)}</strong></div>}
        {data.current_fvg && <div><span>Fair value gap</span><strong>{humanize(data.current_fvg.lifecycle)}</strong></div>}
        {data.ai_missing_sources.length > 0 && <div><span>Missing sources</span><strong>{data.ai_missing_sources.join(', ')}</strong></div>}
        {data.ai_degraded_sources.length > 0 && <div><span>Degraded sources</span><strong>{data.ai_degraded_sources.join(', ')}</strong></div>}
      </div>
    </DetailDrawer>}
  </section>
}

export function QuantForecastSummary({ forecast, calibration, unavailableReason }: { forecast: QuantForecastResult | null; calibration: QuantCalibrationReport | null; unavailableReason?: string }) {
  return <section className="ai-card">
    <SectionHeader eyebrow="Quantitative forecast" title="Multi-horizon outlook" action={<Sigma size={19} />} />
    {!forecast || forecast.status !== 'available' ? <EmptyState title="Forecast not completed" detail={humanize(forecast?.reason_codes[0] ?? unavailableReason ?? 'awaiting_first_completed_cycle')} /> : <>
      <div className="horizon-stack">
        {forecast.predictions.map(item => <article className="horizon" key={item.horizon.horizon_id}>
          <div className="horizon__head"><div><strong>{item.horizon.horizon_id.replaceAll('_', ' × ')}</strong><span>{item.horizon.duration_seconds / 60} minute horizon</span></div><StatusBadge tone="neutral">{humanize(forecast.calibration_status)}</StatusBadge></div>
          <ProbabilityBar buy={item.buy_probability} sell={item.sell_probability} neutral={item.neutral_probability} />
          <div className="horizon__meta"><span>Expected move <strong>{item.expected_base_movement.toFixed(3)}</strong></span><span>Expected volatility <strong>{item.expected_volatility.toFixed(3)}</strong></span></div>
        </article>)}
      </div>
      <div className="card-foot"><span>{humanize(forecast.model_name)} · v{forecast.model_version}</span><StatusBadge tone="neutral">Quantitative model</StatusBadge></div>
      <DetailDrawer label="Model and calibration details">
        <div className="detail-grid">
          <div><span>Provider kind</span><strong>{humanize(forecast.model_kind)}</strong></div>
          <div><span>Feature schema</span><strong>{forecast.feature_schema_version}</strong></div>
          <div><span>Calibration sample</span><strong>{calibration?.sample_count ?? 0}</strong></div>
          {calibration?.brier_score != null && <div><span>Brier score</span><strong>{calibration.brier_score.toFixed(4)}</strong></div>}
        </div>
      </DetailDrawer>
    </>}
  </section>
}

export function AIReasoningSummary({ data, unavailableReason }: { data: AIReasoningDashboard | null; unavailableReason?: string }) {
  const analysis = data?.analysis
  const output = analysis?.output
  return <section className="ai-card">
    <SectionHeader eyebrow="AI deep market analysis" title="Market interpretation" action={<BrainCircuit size={19} />} />
    {!analysis || analysis.status !== 'available' || !output ? <EmptyState title="Analysis not completed" detail={humanize(analysis?.validation_errors[0] ?? unavailableReason ?? 'awaiting_quant_forecast')} /> : <>
      <div className="reasoning-lead">
        <span className={`direction-mark direction-mark--${output.market_regime.classification}`}><Sparkles size={16} />{humanize(output.market_regime.classification)}</span>
        <div><strong>{output.executive_summary}</strong><small>{new Date(analysis.analysis_timestamp).toLocaleString()}</small></div>
      </div>
      <div className="summary-list">
        <div><span>Regime strength</span><strong>{output.market_regime.strength.toFixed(0)}</strong></div>
        <div><span>Analysis confidence</span><strong>{percent(output.analysis_confidence)}</strong></div>
        <div><span>Higher-timeframe bias</span><strong>{humanize(output.higher_timeframe_context.bias)}</strong></div>
        <div><span>Momentum interpretation</span><strong>{humanize(output.momentum_analysis.direction)}</strong></div>
        <div><span>Volatility</span><strong>{humanize(output.volatility_analysis.state)}</strong></div>
        {output.key_risks[0] && <div><span>Primary risk</span><strong>{output.key_risks[0].claim}</strong></div>}
      </div>
      <div className="card-foot"><span>{analysis.provider_metadata.provider} · {analysis.provider_metadata.model}</span><StatusBadge tone="positive">Validated analysis</StatusBadge></div>
      <DetailDrawer label="Analysis evidence and lineage">
        <p className="reasoning-copy">{output.market_structure.recent_change}</p>
        <div className="detail-grid">
          {output.alternative_scenarios.length > 0 && <div><span>Alternative scenarios</span><strong>{output.alternative_scenarios.map(item => `${item.name} ${(item.probability * 100).toFixed(0)}%`).join(', ')}</strong></div>}
          <div><span>Bullish evidence</span><strong>{output.bullish_evidence.length}</strong></div>
          <div><span>Bearish evidence</span><strong>{output.bearish_evidence.length}</strong></div>
          {output.contradictions.length > 0 && <div><span>Contradictions</span><strong>{output.contradictions.length}</strong></div>}
          <div><span>Prompt version</span><strong>{analysis.provider_metadata.prompt_version}</strong></div>
        </div>
      </DetailDrawer>
    </>}
  </section>
}
